#include <cstdio>

#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
static bool sortByMaterial = false;
static bool hasGBuffer = false;
static bool useGBuffer = true;
static ShadeableIntersection* dev_intersections_gbuffer = NULL;
static thrust::device_vector<int> index_array;
static Mesh* dev_meshes = NULL;
static OctreeDev* dev_octrees = NULL;
static int meshSize = 0;
static int numTrees = 0;
static int nStreams = 4;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_intersections_gbuffer, pixelcount * sizeof(ShadeableIntersection));
	
	index_array.resize(pixelcount);
	thrust::sequence(thrust::device, index_array.begin(), index_array.end());

	meshSize = scene->meshes.size();
	cudaMallocManaged(&dev_meshes, meshSize * sizeof(Mesh));
	cudaMemcpy(dev_meshes, scene->meshes.data(), meshSize * sizeof(Mesh), cudaMemcpyHostToDevice);
	for (int i = 0; i < meshSize; i++)
	{
		Mesh& mesh = dev_meshes[i];
		mesh.materialid = scene->meshes[i].materialid;
		mesh.numVertices = scene->meshes[i].numVertices;
		mesh.numIndices = scene->meshes[i].numIndices;
		mesh.boundingVolume = scene->meshes[i].boundingVolume;
		mesh.translation = scene->meshes[i].translation;
		mesh.rotation = scene->meshes[i].rotation;
		mesh.scale = scene->meshes[i].scale;
		mesh.transform = scene->meshes[i].transform;
		mesh.inverseTransform = scene->meshes[i].inverseTransform;
		mesh.invTranspose = scene->meshes[i].invTranspose;

		float* dev_vertices;
		unsigned short* dev_indices;
		cudaMalloc(&dev_vertices, scene->meshes[i].numVertices * sizeof(float) * 3);
		cudaMalloc(&dev_indices, scene->meshes[i].numIndices * sizeof(unsigned short));

		cudaMemcpy(dev_vertices, scene->meshes[i].vertices, scene->meshes[i].numVertices * sizeof(float) * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_indices, scene->meshes[i].indices, scene->meshes[i].numIndices * sizeof(unsigned short), cudaMemcpyHostToDevice);

		dev_meshes[i].vertices = dev_vertices;
		dev_meshes[i].indices = dev_indices;
	}

	numTrees = scene->octrees.size();
	cudaMallocManaged(&dev_octrees, numTrees * sizeof(OctreeDev));
	cudaMemcpy(dev_octrees, scene->octrees.data(), numTrees * sizeof(OctreeDev), cudaMemcpyHostToDevice);
	
	for (int i = 0; i < numTrees; i++) {
		OctreeDev& octree = dev_octrees[i];
		octree.root = scene->octrees[i].root;
		octree.materialid = scene->octrees[i].materialid;
		octree.numNodes = scene->octrees[i].nodes.size();
		octree.transform = scene->octrees[i].transform;
		octree.inverseTransform = scene->octrees[i].inverseTransform;
		octree.invTranspose = scene->octrees[i].invTranspose;

		OctreeNode* dev_nodes;
		Triangle* dev_triangles;
		Geom* dev_bounds;
		int* dev_dataStarts;

		cudaMalloc(&dev_nodes, octree.numNodes * sizeof(OctreeNode));
		cudaMalloc(&dev_triangles, scene->octrees[i].triangles.size() * sizeof(Triangle));
		cudaMalloc(&dev_bounds, scene->octrees[i].boundingBoxes.size() * sizeof(Geom));
		cudaMalloc(&dev_dataStarts, scene->octrees[i].dataStarts.size() * sizeof(int));

		cudaMemcpy(dev_nodes, scene->octrees[i].nodes.data(), octree.numNodes * sizeof(OctreeNode), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_triangles, scene->octrees[i].triangles.data(), scene->octrees[i].triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bounds, scene->octrees[i].boundingBoxes.data(), scene->octrees[i].boundingBoxes.size() * sizeof(Geom), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_dataStarts, scene->octrees[i].dataStarts.data(), scene->octrees[i].dataStarts.size() * sizeof(int), cudaMemcpyHostToDevice);

		dev_octrees[i].nodes = dev_nodes;
		dev_octrees[i].triangles = dev_triangles;
		dev_octrees[i].boundingBoxes = dev_bounds;
		dev_octrees[i].dataStarts = dev_dataStarts;
	}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	checkCUDAError("pathtraceFree");
	cudaFree(dev_paths);
	checkCUDAError("pathtraceFree");
	cudaFree(dev_geoms);
	checkCUDAError("pathtraceFree");
	cudaFree(dev_materials);
	checkCUDAError("pathtraceFree");
	cudaFree(dev_intersections);
	checkCUDAError("pathtraceFree");
	cudaFree(dev_intersections_gbuffer);
	checkCUDAError("pathtraceFree");
	for (int i = 0; i < meshSize; i++)
	{
		cudaFree(dev_meshes[i].vertices);
		cudaFree(dev_meshes[i].indices);
	}
	cudaFree(dev_meshes);
	for (int i = 0; i < numTrees; i++) {
		cudaFree(dev_octrees[i].nodes);
		cudaFree(dev_octrees[i].triangles);
		cudaFree(dev_octrees[i].boundingBoxes);
		cudaFree(dev_octrees[i].dataStarts);
	}
	cudaFree(dev_octrees);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, Mesh* meshes
	, OctreeDev* octrees
	, int geoms_size
	, int meshes_size
	, int num_trees
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		int hit_mesh_index = -1;
		int hit_tree_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}
		// for (int i = 0; i < meshes_size; i++)
		// {
		// 	Mesh& mesh = meshes[i];

		// 	t = boxIntersectionTest(mesh.boundingVolume, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		// 	if (t <= 0.0f)
		// 	{
		// 		continue;
		// 	}
		// 	t = meshIntersectionTest(mesh, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		// 	// if (t > 0.0f && t_min > t)
		// 	// {
		// 	// 	t_min = t;
		// 	// 	hit_mesh_index = i;
		// 	// 	hit_geom_index = -1;
		// 	// 	hit_tree_index = -1;
		// 	// 	intersect_point = tmp_intersect;
		// 	// 	normal = tmp_normal;
		// 	// }
		// }
		for (int i = 0; i < num_trees; i++)
		{
			OctreeDev& octree = octrees[i];
			//t = boxIntersectionTest(octree.boundingBoxes[3], pathSegment.ray, tmp_intersect, tmp_normal, outside);
			t = octreeIntersectionTest(octree, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_mesh_index = -1;
				hit_geom_index = -1;
				hit_tree_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1 && hit_mesh_index == -1 && hit_tree_index == -1)
		{
			intersections[path_index].t = -1.0f;
			return;
		}
		intersections[path_index].t = t_min;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].intersectPoint = intersect_point;
		if (hit_mesh_index != -1)
		{
			intersections[path_index].materialId = meshes[hit_mesh_index].materialid;
		}
		if (hit_geom_index != -1)
		{
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		}
		if (hit_tree_index != -1)
		{
			intersections[path_index].materialId = octrees[hit_tree_index].materialid;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeMaterial(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				scatterRay(pathSegments[idx], intersection.intersectPoint, intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;
			}
		}
		else {
			pathSegments[idx].color = BACKGROUND_COLOR;
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int offset)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x + offset;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct ExtractMaterialId
{
    __host__ __device__
    int operator()(const ShadeableIntersection& intersection) const {
        return intersection.materialId;
    }
};


struct is_valid {
	__host__ __device__
		bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaStream_t streams[nStreams];
	for (int i = 0; i < nStreams; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / (blockSize2d.y));

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d, 0>> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	// generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	// checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	bool iterationComplete = false;
	hasGBuffer = !(iter == 1) && useGBuffer;

	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// tracing
		if (!hasGBuffer || depth > 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
						depth
						, num_paths
						, dev_paths
						, dev_geoms
						, dev_meshes
						, dev_octrees
						, hst_scene->geoms.size()
						, hst_scene->meshes.size()
						, hst_scene->octrees.size()
						, dev_intersections
						);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();

			if (!hasGBuffer && useGBuffer) {
				cudaMemcpy(dev_intersections_gbuffer, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				
				hasGBuffer = true;
				checkCUDAError("cudaMemcpy");
			}			
		}
		else {
			cudaMemcpy(dev_intersections, dev_intersections_gbuffer, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}	

		checkCUDAError("trace one bounce");
		//cudaDeviceSynchronize();
		depth++;

		if (sortByMaterial)
		{
			std::cout << "sortByMaterial" << std::endl;
			thrust::device_ptr<PathSegment> dev_paths_ptr(dev_paths);
			thrust::device_ptr<ShadeableIntersection> dev_intersections_ptr(dev_intersections);

			thrust::device_vector<int> materialIds(num_paths);
			thrust::transform(dev_intersections_ptr, dev_intersections_ptr + num_paths, materialIds.begin(), ExtractMaterialId());

			thrust::sort_by_key(thrust::device, materialIds.begin(), materialIds.end(),
								thrust::make_zip_iterator(thrust::make_tuple(dev_paths_ptr, dev_intersections_ptr)));

			checkCUDAError("thrust::sort_by_key");
		}

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d, blockSize1d * sizeof(ShadeableIntersection) >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths, 
			dev_materials
		    );
		checkCUDAError("shadeMaterial");

		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, is_valid());
		num_paths = dev_path_end - dev_paths;

		if (num_paths <= 0 || depth >= traceDepth) {
			iterationComplete = true;
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// cudaDeviceSynchronize();
	// int streamSize = pixelcount / nStreams;
	// dim3 numBlocksPixels = (streamSize + blockSize1d - 1) / blockSize1d;
	// for (int i = 0; i < nStreams; i++)
	// {
	// 	int offset = i * streamSize;
	// 	finalGather << <numBlocksPixels, blockSize1d, 0, streams[i] >> > (num_paths, dev_image, dev_paths, offset);
	// }
	// // Assemble this iteration and apply it to the image
	// cudaDeviceSynchronize();
	// checkCUDAError("finalGather");

	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths,0);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	for (int i = 0; i < nStreams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}

	checkCUDAError("pathtrace");
}
