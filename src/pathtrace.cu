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

#define SORT_MATERIALS 0
#define FIRST_BOUNCE_CACHE 0
#define ANTI_ALIASING 0

#define NAIVE 0
#define DIRECT_MIS 0
#define FULL 1

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

struct isPathValid {
	__host__ __device__
		bool operator()(const PathSegment& x)
	{
		return x.remainingBounces > 0;
	}
};

struct compareByMaterialId {
	__host__ __device__
		bool operator()(const ShadeableIntersection& x, const ShadeableIntersection& y)
	{
		return x.materialId < y.materialId;
	}
};

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
static Triangle* dev_triangles = NULL;
static Light* dev_lights = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

#if FIRST_BOUNCE_CACHE
static ShadeableIntersection* dev_first_bounce_cache = NULL;
#endif


// TODO: static variables for device memory, any extra info you need, etc
// ...

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
#if FIRST_BOUNCE_CACHE
	cudaMalloc(&dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce_cache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

	// lights
	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	// triangles
	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created

#if FIRST_BOUNCE_CACHE
	cudaFree(dev_first_bounce_cache);
#endif

	cudaFree(dev_lights);
	cudaFree(dev_triangles);

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

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.isFromCamera = true;
		segment.isSpecularBounce = false;

		// anti-aliasing by jittering
		// u01(rng) - 0.5f
		glm::vec2 bias(0.0f);
#if ANTI_ALIASING
		bias = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
#endif
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + bias.x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + bias.y - (float)cam.resolution.y * 0.5f)
		);

		// Physically-based depth-of-field
		if (cam.lensRadius > 0.0f) {
			// generate random point on lens
			glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
			glm::vec2 pLens = cam.lensRadius * concentricSampleDisk(sample);

			// compute point on plane of focus
			float ft = glm::abs(cam.focalDistance / segment.ray.direction.z);
			glm::vec3 pFocus = segment.ray.origin + ft * segment.ray.direction;

			// update ray for effect of lens
			segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__device__ bool boundingBoxTest(const Ray* ray, const Geom* geom) {
	float min_t = EPSILON;
	float max_t = 1000.f;

	for (int a = 0; a < 3; a++) {
		float o = ray->origin[a];
		float invD = 1.0f / ray->direction[a];

		float t0 = (geom->minPoint[a] - o) * invD;
		float t1 = (geom->maxPoint[a] - o) * invD;

		if (invD < 0) {
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}

		min_t = fmax(t0, min_t);
		max_t = fmin(t1, max_t);

		if (max_t <= min_t) {
			return false;
		}
	}

	return true;
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
	, int geoms_size
	, Triangle* triangles
	, int triangles_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		computeRayIntersection(geoms, geoms_size, triangles, triangles_size, pathSegment.ray, intersections[path_index]);
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

// naive path tracing
__global__ void shadeNaive(
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
		PathSegment& cur = pathSegments[idx];
		if (intersection.t > 0.0) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				cur.color += cur.throughput * (materialColor * material.emittance);
				cur.remainingBounces = 0; // terminate path
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				scatterRay(cur,
					getPointOnRay(cur.ray, intersection.t),
					intersection.surfaceNormal, intersection.surfaceTangent, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		} else {
			cur.color = glm::vec3(0.0f);
			cur.remainingBounces = 0; // terminate path
		}
	}
}

// multi importance sampling
__global__ void shadeDirectMIS(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* geoms
	, int numGeoms
	, Triangle* triangles
	, int numTriangles
	, Light* lights
	, int numLights
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& cur = pathSegments[idx];

		if (intersection.t > 0.0) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;
			glm::vec3 point = getPointOnRay(cur.ray, intersection.t);

			if (material.emittance > 0.0f) {
				cur.color = (materialColor * material.emittance);
				cur.remainingBounces = 0; // terminate path
			} else {
				cur.color = sampleUniformLight(point, intersection.surfaceNormal, intersection.surfaceTangent, cur.ray.direction
					, materials, material, geoms, numGeoms, triangles, numTriangles, lights, numLights, rng);
				cur.remainingBounces = 0;
			}
		} else {
			cur.color = glm::vec3(0.0f);
			cur.remainingBounces = 0; // terminate path
		}
	}
}

// full light integrator
__global__ void shadeFull(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* geoms
	, int numGeoms
	, Triangle* triangles
	, int numTriangles
	, Light* lights
	, int numLights
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& cur = pathSegments[idx];
		if (intersection.t > 0.0) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			Material material = materials[intersection.materialId];
			glm::vec3 point = getPointOnRay(cur.ray, intersection.t);
			glm::vec3 materialColor = material.color;

			// specular or direct from camera
			if (cur.isFromCamera || cur.isSpecularBounce) {
				// If the material indicates that the object was a light, "light" the ray
				cur.color += cur.throughput * materialColor * material.emittance;
				
				cur.isFromCamera = false;
				cur.isSpecularBounce = false;
			} 


			// If the surface is diffuse or microfacet, compute MIS direct light
			if (material.type == MaterialType::DIFFUSE) {
				glm::vec3 Ld = sampleUniformLight(point, intersection.surfaceNormal, intersection.surfaceTangent, cur.ray.direction
					, materials, material, geoms, numGeoms, triangles, numTriangles, lights, numLights, rng);
				cur.color += Ld * cur.throughput;
			}
				
			// Sample BSDF
			scatterRay(cur, point, intersection.surfaceNormal, intersection.surfaceTangent, material, rng);

			// TODO: subsurface scattering

			// TODO: russian roulette
		}
		else {
			cur.color = glm::vec3(0.0f);
			if (cur.isFromCamera || cur.isSpecularBounce) {
				// TODO: environment map
				/*for (int i = 0; i < numLights; ++i) {
					Material mat = materials[lights[i].geom.materialid];
					cur.color += cur.throughput * mat.color * mat.emittance;
				}*/
			}
			cur.remainingBounces = 0; // terminate path
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(32, 32);
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

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if FIRST_BOUNCE_CACHE
		if (iter == 1 || depth != 0) {

			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_triangles
				, hst_scene->triangles.size()
				, dev_intersections
				);

			if (depth == 0) {
				// cache
				cudaMemcpy(dev_first_bounce_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		} else {
			cudaMemcpy(dev_intersections, dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_triangles
			, hst_scene->triangles.size()
			, dev_intersections
			);
#endif

		depth++;

		// sort by material?
#if SORT_MATERIALS
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + cur_num_paths, dev_paths, compareByMaterialId());
#endif
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if NAIVE
		shadeNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
#endif
#if DIRECT_MIS
		shadeDirectMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_triangles,
			hst_scene->triangles.size(),
			dev_lights,
			hst_scene->lights.size()
			);
#endif
#if FULL
		shadeFull << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_triangles,
			hst_scene->triangles.size(),
			dev_lights,
			hst_scene->lights.size()
			);
#endif

		// remove rays with zero remaining bounces
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, isPathValid());
		num_paths = dev_path_end - dev_paths;

		iterationComplete = (depth >= traceDepth || num_paths <= 0); // TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
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
