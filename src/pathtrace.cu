#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MATERIALSORT 0
#define FIRSTBOUNCECACHE 0

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
#if FIRSTBOUNCECACHE
static ShadeableIntersection* dev_fb_int_cache = NULL;
static bool fb_cached = false;
#endif

//gltf meshes
static Triangle* dev_mesh_tris = NULL;

static ImageInfo* dev_mesh_img_infos = NULL;
static glm::vec3* dev_mesh_img_data = NULL;

#if BVH
//bvh tree structs
static BVHNode* dev_bvh_scene_tree = NULL;
static BVHTriIndex* dev_bvh_tri_indices = NULL;
static BVHGeomIndex* dev_bvh_geom_indices = NULL;
#endif

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

#if FIRSTBOUNCECACHE
	cudaMalloc(&dev_fb_int_cache, pixelcount * sizeof(ShadeableIntersection));
#endif
	cudaMalloc(&dev_mesh_tris, scene->mesh_triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_mesh_tris, scene->mesh_triangles.data(), scene->mesh_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_mesh_img_infos, scene->image_infos.size() * sizeof(ImageInfo));
	cudaMemcpy(dev_mesh_img_infos, scene->image_infos.data(), scene->image_infos.size() * sizeof(ImageInfo), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_mesh_img_data, scene->image_data.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_mesh_img_data, scene->image_data.data(), scene->image_data.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

#if BVH
	cudaMalloc(&dev_bvh_scene_tree, scene->bvh_nodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvh_scene_tree, scene->bvh_nodes.data(), scene->bvh_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_tri_indices, scene->bvh_tri_indices.size() * sizeof(BVHTriIndex));
	cudaMemcpy(dev_bvh_tri_indices, scene->bvh_tri_indices.data(), scene->bvh_tri_indices.size() * sizeof(BVHTriIndex), cudaMemcpyHostToDevice);


	cudaMalloc(&dev_bvh_geom_indices, scene->bvh_geom_indices.size() * sizeof(BVHGeomIndex));
	cudaMemcpy(dev_bvh_geom_indices, scene->bvh_geom_indices.data(), scene->bvh_geom_indices.size() * sizeof(BVHGeomIndex), cudaMemcpyHostToDevice);
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_mesh_tris);
	cudaFree(dev_mesh_img_infos);
	cudaFree(dev_mesh_img_data);
	
#if FIRSTBOUNCECACHE
	cudaFree(dev_fb_int_cache);
#endif

#if BVH
	cudaFree(dev_bvh_scene_tree);
	cudaFree(dev_bvh_tri_indices);
	cudaFree(dev_bvh_geom_indices);
#endif
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
#if BVH
__global__ void computeIntersectionsBvh(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles
	, ImageInfo* img_infos
	, glm::vec3* img_data
	, BVHNode* bvh_tree
	, BVHTriIndex* bvh_tri_indices
	, BVHGeomIndex* bvh_geom_indices
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		glm::vec3 int_bary;
		int tri_index = -1;

		t_min = bvh_intersection_test(pathSegment.ray, intersect_point, normal, outside,
			triangles, geoms, img_infos, img_data, int_bary, tri_index, hit_geom_index,
			bvh_tree, bvh_tri_indices, bvh_geom_indices);

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].geom = &geoms[hit_geom_index];
			//if tri note tri index and bary coords
			if (tri_index >= 0) {
				intersections[path_index].int_tri = &triangles[tri_index];
				intersections[path_index].bary = int_bary;
			}
		}
	}
}
#else
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles
	, ImageInfo* img_infos
	, glm::vec3* img_data
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
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec3 int_bary;
		int tri_index;
		int geom_index;

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
			else if (geom.type == MESH_PRIM)
			{
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, triangles, img_infos, img_data, int_bary, tri_index);
			}

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

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].geom = &geoms[hit_geom_index];
			if (tri_index >= 0) {
				intersections[path_index].int_tri = &triangles[tri_index];
				intersections[path_index].bary = int_bary;
			}
		}
	}
}
#endif
// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, ImageInfo* img_infos
	, glm::vec3* img_data
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& currSegment = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			//if the intersected geom has a texture, use that color, else its material
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// if int geom has tex overwrite mat color
			// FIXME better integration here with mat
			if (intersection.geom->texture_index >= 0) {
				glm::vec3 int_bary = intersection.bary;
				Triangle min_tri = *intersection.int_tri;
				glm::vec2 mesh_uv = int_bary.z * min_tri.points[0].tex_uv + int_bary.x * min_tri.points[1].tex_uv + int_bary.y * min_tri.points[2].tex_uv;
				ImageInfo tex_map = img_infos[intersection.geom->texture_index];
				glm::ivec2 sample_uv = mesh_uv * glm::vec2(tex_map.img_w, tex_map.img_h);
				materialColor = img_data[tex_map.data_start_index + (sample_uv[1] * tex_map.img_w + sample_uv[0])];
			}
			else if (intersection.geom->base_color != glm::vec3(-1.f)) {
				materialColor = intersection.geom->base_color;
			}


			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				currSegment.color *= (materialColor * material.emittance);
				currSegment.remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else {
				currSegment.color *= materialColor;
				currSegment.remainingBounces--;
				//scatter to gen new ray
				scatterRay(currSegment, getPointOnRay(currSegment.ray, intersection.t), intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			currSegment.color = glm::vec3(0.0f);
			currSegment.remainingBounces = 0;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

// predicate for path compaction/partition
struct nzero_bounces
{
	__host__ __device__
		inline bool operator()(const PathSegment& p) 
	{
		return p.remainingBounces != 0;
	}
};

struct mat_int_ordering
{
	__host__ __device__
		inline bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) 
	{
		return i1.materialId < i2.materialId;
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

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if FIRSTBOUNCECACHE
		if (fb_cached && depth == 0) {
			cudaMemcpy(dev_intersections, dev_fb_int_cache, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
#endif
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			// tracing
#if BVH
			computeIntersectionsBvh << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_mesh_tris
				, dev_mesh_img_infos
				, dev_mesh_img_data
				, dev_bvh_scene_tree
				, dev_bvh_tri_indices
				, dev_bvh_geom_indices
				);
#else
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_mesh_tris
				, dev_mesh_img_infos
				, dev_mesh_img_data
				);
#endif

			checkCUDAError("trace one bounce");

#if FIRSTBOUNCECACHE
		}
		if (!fb_cached && depth == 0) {
			cudaMemcpy(dev_fb_int_cache, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			fb_cached = true;
		}
#endif
		cudaDeviceSynchronize();
		depth++;

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
	  

		// sort dev_paths by material id, shuffle dev_intersections in same way
#if MATERIALSORT
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, mat_int_ordering());
#endif

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_mesh_img_infos,
			dev_mesh_img_data
			);

		//compact and reduce numpaths
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, nzero_bounces());
		num_paths = dev_path_end - dev_paths;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}

		if (num_paths == 0) iterationComplete = true;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
