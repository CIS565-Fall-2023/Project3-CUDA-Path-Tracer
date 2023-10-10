#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
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

#define CACHE_FIRST_BOUNCE 1
#define SORT_MATERIAL 1
#define STREAM_COMPACTION 1

#define DEPTH_OF_FIELD 0
#define ANTIALIASING 0
#define DIRECT_LIGHTING 1
#define MOTION_BLUR 0
#define MOTION_VELO glm::vec3(0.5, 0.5, 0.0)
#define BVH 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

static Scene* hst_scene = NULL;

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	if (hst_scene) 
	{
		fprintf(stderr, " (%d)", hst_scene->triangles.size());
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

static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_first_bounce = NULL;
static Geom* dev_lights = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvh_nodes = NULL;
static Texture* dev_textures = NULL;
static cudaTextureObject_t* host_texObjects = NULL;
static cudaTextureObject_t* dev_texObjects = NULL;
static cudaArray_t* host_texData = NULL;
static int numTexture = 0;

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
	cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_bvh_nodes, scene->bvhNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvh_nodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	initTexture();
	
	checkCUDAError("pathtraceInit");
}

void initTexture() 
{
	if (!hst_scene) return;
	numTexture = hst_scene->textures.size();
	cudaMalloc(&dev_textures, numTexture * sizeof(Texture));
	cudaMemcpy(dev_textures, hst_scene->textures.data(), numTexture * sizeof(Texture), cudaMemcpyHostToDevice);
	if (numTexture > 0) 
	{
		host_texObjects = new cudaTextureObject_t[numTexture];
		host_texData = new cudaArray_t[numTexture];

		cudaChannelFormatDesc channelDesc;

		for (int i = 0; i < numTexture; i++) 
		{
			const Texture& tex = hst_scene->textures[i];

			channelDesc = cudaCreateChannelDesc<float4>();

			cudaMallocArray(&host_texData[i], &channelDesc, tex.width, tex.height);
			cudaMemcpy2DToArray(host_texData[i], 0, 0, tex.data, tex.width * sizeof(float4), tex.width * sizeof(float4), tex.height, cudaMemcpyHostToDevice);
		
			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = host_texData[i];

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&host_texObjects[i], &resDesc, &texDesc, NULL);
		}
		cudaMalloc((void**)&dev_texObjects, numTexture * sizeof(cudaTextureObject_t));
		cudaMemcpy(dev_texObjects, host_texObjects, numTexture * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	}
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_first_bounce);
	cudaFree(dev_lights);
	cudaFree(dev_triangles);
	cudaFree(dev_bvh_nodes);

	cudaFree(dev_textures);

	for (int i = 0; i < numTexture; i++) 
	{
		cudaDestroyTextureObject(host_texObjects[i]);
		cudaFreeArray(host_texData[i]);
	}

	delete[] host_texObjects;
	delete[] host_texData;

	checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 concentricSampleDisk(const glm::vec2& u) 
{
	glm::vec2 uOffset = 2.0f * u - glm::vec2(1.f, 1.f);
	if (uOffset.x == 0.0 && uOffset.y == 0.0) 
	{
		return glm::vec2(0, 0);
	}
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) 
	{
		r = uOffset.x;
		theta = PI / 4.f * (uOffset.y / uOffset.x);
	}
	else 
	{
		r = uOffset.y;
		theta = PI / 2.f - PI / 4.f * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cos(theta), std::sin(theta));
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
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

#if MOTION_BLUR
		glm::vec3 jiiterVec = u01(rng) * MOTION_VELO;
		segment.ray.origin += jiiterVec;
#endif

#if ANTIALIASING && !CACHE_FIRST_BOUNCE
		float jitterX = u01(rng);
		float jitterY = u01(rng);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

#if DEPTH_OF_FIELD
		if (cam.lensRadius > 0.0) 
		{
			glm::vec2 pLens = cam.lensRadius * concentricSampleDisk(glm::vec2(u01(rng), u01(rng)));
			float ft = glm::abs(cam.focalDistance / segment.ray.direction.z);
			glm::vec3 pFoucs = segment.ray.origin + ft * segment.ray.direction;

			segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
			segment.ray.direction = glm::normalize(pFoucs - segment.ray.origin);
		}
#endif
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
	, Triangle* triangles
	, BVHNode* bvhNodes
	, int tri_size
	, int geoms_size
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
		glm::vec2 uv;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

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
			else if (geom.type == OBJ) 
			{
#if BVH
				t = bvhTriangleIntersectionTest(geom, pathSegment.ray, triangles, bvhNodes, tri_size, tmp_intersect, tmp_normal, tmp_uv, outside);
#else
				t = triangleIntersectionTest(geom, pathSegment.ray, triangles, tri_size, tmp_intersect, tmp_normal, tmp_uv, outsides);
#endif
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
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
			intersections[path_index].texId = geoms[hit_geom_index].texId;
			intersections[path_index].uv = uv;
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
__global__ void kernShadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, cudaTextureObject_t* texObjects
	, Texture* textures
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment& curSeg = pathSegments[idx];
		if (curSeg.remainingBounces <= 0) return;

		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, curSeg.remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				curSeg.color *= (materialColor * material.emittance);
				curSeg.remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				scatterRay(curSeg, getPointOnRay(curSeg.ray, intersection.t),
						intersection.surfaceNormal, intersection.uv, material, rng, texObjects, textures, intersection.texId);
				curSeg.remainingBounces--;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			curSeg.color = glm::vec3(0.0f);
			curSeg.remainingBounces = 0;
		}
	}
}

__global__ void kernShadeMaterialDirectLight(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* lights
	, int num_lights
	, cudaTextureObject_t* texObjects
	, Texture* textures
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment& curSeg = pathSegments[idx];
		if (curSeg.remainingBounces <= 0) return;

		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, curSeg.remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				curSeg.color *= (materialColor * material.emittance);
				curSeg.remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				scatterRay(curSeg, getPointOnRay(curSeg.ray, intersection.t),
					intersection.surfaceNormal, intersection.uv, material, rng, texObjects, textures, intersection.texId);

				if (curSeg.remainingBounces == 1) 
				{
					thrust::uniform_real_distribution<int> u02(0, num_lights - 1);
					glm::vec3 lightPos = glm::vec3(lights[u02(rng)].transform * glm::vec4(u01(rng), u01(rng), u01(rng), 1.0f));
					curSeg.ray.direction = glm::normalize(lightPos - curSeg.ray.origin);
				}

				curSeg.remainingBounces--;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			curSeg.color = glm::vec3(0.0f);
			curSeg.remainingBounces = 0;
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

struct materialComparator 
{
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) 
	{
		return a.materialId < b.materialId;
	}
};

struct zeroBounce 
{
	__host__ __device__ bool operator()(const PathSegment& pseg)
	{
		return pseg.remainingBounces > 0;
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

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if  CACHE_FIRST_BOUNCE

		if (depth == 0 && iter == 1) 
		{
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_triangles
				, dev_bvh_nodes
				, hst_scene->triangles.size()
				, hst_scene->geoms.size()
				, dev_first_bounce
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_first_bounce, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0 && iter != 1) 
		{
			cudaMemcpy(dev_intersections, dev_first_bounce, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_triangles
				, dev_bvh_nodes
				, hst_scene->triangles.size()
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, dev_triangles
			, dev_bvh_nodes
			, hst_scene->triangles.size()
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if  SORT_MATERIAL
		thrust::device_ptr<ShadeableIntersection> isects(dev_intersections);
		thrust::device_ptr<PathSegment> psegs(dev_paths);
		thrust::sort_by_key(isects, isects + num_paths, psegs, materialComparator());	
#endif

#if DIRECT_LIGHTING
		kernShadeMaterialDirectLight<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_lights,
			hst_scene->lights.size(),
			dev_texObjects,
			dev_textures
			);
#else
		kernShadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_texObjects,
			dev_textures
			);
#endif

#if STREAM_COMPACTION
		PathSegment* new_path_end = thrust::stable_partition(thrust::device, dev_paths,
			dev_paths + num_paths, zeroBounce());
		num_paths = new_path_end - dev_paths;
		if (num_paths <= 0) iterationComplete = true;
#else
		iterationComplete = true; // TODO: should be based off stream compaction results.
#endif

		num_paths = dev_path_end - dev_paths;
		
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
