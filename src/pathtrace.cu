#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/projection.hpp"
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

struct is_traceable {
	__host__ __device__
		bool operator()(const PathSegment p)
	{
		return p.remainingBounces > 0;
	}
};

struct no_color {
	__host__ __device__
		bool operator()(const PathSegment p)
	{
		return p.color == glm::vec3(0.f);
	}
};

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
// ...
static thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections = NULL;
static thrust::device_ptr<PathSegment> dev_thrust_paths = NULL;
static ShadeableIntersection* dev_fbc = NULL;
static thrust::device_ptr<ShadeableIntersection> dev_thrust_fbc = NULL;

static Triangle* dev_prims = NULL;
static BVHNode* dev_bvh = NULL;

static int numTextures = 0;
static cudaTextureObject_t* host_textureObjs = NULL;
static cudaTextureObject_t* dev_textureObjs = NULL;
static cudaArray_t* host_textureDataPtrs = NULL;

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
	dev_thrust_intersections = thrust::device_ptr<ShadeableIntersection>(dev_intersections);
	dev_thrust_paths = thrust::device_ptr<PathSegment>(dev_paths);

	cudaMalloc(&dev_fbc, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_fbc, 0, pixelcount * sizeof(ShadeableIntersection));
	dev_thrust_fbc = thrust::device_ptr<ShadeableIntersection>(dev_fbc);

	cudaMalloc(&dev_prims, scene->prims.size() * sizeof(Triangle));
	cudaMemcpy(dev_prims, scene->prims.data(), scene->prims.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh, scene->BVHNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvh, scene->BVHNodes.data(), scene->BVHNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	// textures
	numTextures = scene->textures.size();
	if (numTextures > 0) {
		host_textureObjs = new cudaTextureObject_t[numTextures];
		host_textureDataPtrs = new cudaArray_t[numTextures];

		for (int i = 0; i < numTextures; i++) {
			const Texture& tex = scene->textures[i];
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
			cudaMallocArray(&host_textureDataPtrs[i], &channelDesc, tex.width, tex.height);
			cudaMemcpy2DToArray(host_textureDataPtrs[i], 0, 0, tex.host_buffer, tex.width * sizeof(float4), tex.width * sizeof(float4), tex.height, cudaMemcpyHostToDevice);

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = host_textureDataPtrs[i];

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&host_textureObjs[i], &resDesc, &texDesc, NULL);
		}

		cudaMalloc((void**)&dev_textureObjs, numTextures * sizeof(cudaTextureObject_t));
		cudaMemcpy(dev_textureObjs, host_textureObjs, numTextures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	checkCUDAError("before pathtraceFree");
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_fbc);
	cudaFree(dev_prims);
	cudaFree(dev_bvh);
	if (numTextures > 0) {
		cudaFree(dev_textureObjs);
		for (int i = 0; i < numTextures; i++) {
			cudaDestroyTextureObject(host_textureObjs[i]);
			cudaFreeArray(host_textureDataPtrs[i]);
		}
		delete[] host_textureDataPtrs;
		delete[] host_textureObjs;
	}
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

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		glm::vec3 baseDirection = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
		);

		if (cam.lensRadius > 0.f) {
			float ft = cam.focalDistance / glm::length(glm::proj(baseDirection, cam.view));
			glm::vec3 pFocus = cam.position + (baseDirection * ft);
			glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));
			segment.ray.origin = cam.position + (pLens.x * cam.right + pLens.y * cam.up);
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
		}
		else {
			segment.ray.origin = cam.position;
			segment.ray.direction = baseDirection;
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth, 
	int num_paths, PathSegment* pathSegments, 
	Geom* geoms, int geoms_size, 
	Triangle* prims,
	BVHNode* bvh,
	ShadeableIntersection* intersections 
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
		int tmp_primIdx = -1;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, bvh, prims, pathSegment.ray, tmp_normal, tmp_uv, tmp_primIdx, t_min);
			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
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
			intersections[path_index].surfaceNormal = normal;
			if (tmp_primIdx != -1) {
				intersections[path_index].materialId = prims[tmp_primIdx].materialid;
				intersections[path_index].uv = tmp_uv;
			} else {
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			}
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

__global__ void shadeMaterialNaive(
	int iter, 
	int depth, 
	int num_paths, 
	ShadeableIntersection* shadeableIntersections, 
	PathSegment* pathSegments, 
	Material* materials,
	cudaTextureObject_t* texObjs
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment segment = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

			Material& material = materials[intersection.materialId];

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				segment.color *= (material.color * material.emittance);
				segment.remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				scatterRay(segment, getPointOnRay(segment.ray, intersection.t), intersection, material, texObjs, rng);
				if (--segment.remainingBounces < 1) {
					segment.color = glm::vec3(0.f);
				}
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			segment.color = glm::vec3(0.0f);
			segment.remainingBounces = 0;
		}
		pathSegments[idx] = segment;
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		/*if (isnan(iterationPath.color.x) || isnan(iterationPath.color.y) || isnan(iterationPath.color.z))
		{
			image[iterationPath.pixelIndex] = glm::vec3(1, 0, 1);
		}
		else
		{
			image[iterationPath.pixelIndex] += iterationPath.color;
		}*/
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

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
	thrust::device_ptr<PathSegment> dev_thrust_paths_end;
	int num_paths = pixelcount;
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		if (guiData->cacheFirstBounce && guiData->fbCached) {
			thrust::copy(dev_thrust_fbc, dev_thrust_fbc + pixelcount, dev_thrust_intersections);
			checkCUDAError("load first bounce cache");
		}
		else {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths,
				dev_geoms, hst_scene->geoms.size(),
				dev_prims, dev_bvh,
				dev_intersections
			);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		depth++;

		if (guiData->sortMaterial) {
			thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths);
			checkCUDAError("sort by material");
		}

		if (guiData->cacheFirstBounce && !(guiData->fbCached)) {
			thrust::copy(dev_thrust_intersections, dev_thrust_intersections + pixelcount, dev_thrust_fbc);
			checkCUDAError("compute first bounce cache");
			guiData->fbCached = false;
		}

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shadeMaterialNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_textureObjs
			);
		checkCUDAError("Naive Shading");
		// partition rays so that those terminated are not rerun next round
		dev_thrust_paths_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, is_traceable());
		checkCUDAError("partition terminated rays to back");
		num_paths = dev_thrust_paths_end - dev_thrust_paths;
		numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		iterationComplete = num_paths == 0; // TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// remove paths that make no contribution
	dev_thrust_paths_end = thrust::remove_if(dev_thrust_paths, dev_thrust_paths + pixelcount, no_color());
	checkCUDAError("remove no contribution paths");
	num_paths = dev_thrust_paths_end - dev_thrust_paths;
	/*std::cout << num_paths << std::endl;*/
	// Assemble this iteration and apply it to the image
	if (num_paths > 0) {
		dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
		checkCUDAError("finalgather");
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	checkCUDAError("sendPBO");

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
