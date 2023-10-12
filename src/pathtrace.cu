#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <iostream>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtc/matrix_inverse.hpp"
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

// returns true if a path still has bounces left
struct is_ray_done
{
	__host__ __device__
		bool operator()(const PathSegment& path)
	{
		return path.remainingBounces > 0;
	}
};

// compares the material ids of two materials to sort them in ascending order
struct compare_mat_id
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& isect1, ShadeableIntersection& isect2)
	{
		return isect1.materialId < isect2.materialId;
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
static ShadeableIntersection* dev_intersections_first_bounce = NULL;
static Triangle* dev_tris = NULL;
static BVHNode* dev_bvhNodes = NULL;
static int* dev_bvhTriIndices = NULL;

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
#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_intersections_first_bounce, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
#endif
	//if (scene->meshCount > 0) {
		cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
		cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
#if BVH
		cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
		cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_bvhTriIndices, scene->triangles.size() * sizeof(int));
		cudaMemcpy(dev_bvhTriIndices, scene->triIndices.data(), scene->triangles.size() * sizeof(int), cudaMemcpyHostToDevice);
#endif
	//}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_intersections_first_bounce);
#endif
	cudaFree(dev_tris);
#if BVH
	cudaFree(dev_bvhNodes);
	cudaFree(dev_bvhTriIndices);
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
		segment.color = glm::vec3(0.0f);
		segment.throughput = glm::vec3(1.0f);

		// Implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
		thrust::uniform_real_distribution<float> u01(0, 1);
#if AA
		float jitterX = u01(rng);
		float jitterY = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)(x + jitterX) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)(y + jitterY) - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif 

#if DOF
		// Sample point on lens
		glm::vec3 point = concentricSampleDisk(glm::vec2(u01(rng), u01(rng))) * LENS_RADIUS;

		glm::vec3 ref = cam.position + (cam.view * FOCAL_DIST);

		float aspectRatio = ((float)cam.resolution.x / cam.resolution.y);
		float angle = glm::radians(cam.fov.y);
		glm::vec3 V = cam.up * FOCAL_DIST * tan(angle);
		glm::vec3 H = cam.right * FOCAL_DIST * aspectRatio * tan(angle);

		float ndcX = 1.f - ((float)x / cam.resolution.x) * 2.f;
		float ndcY = 1.f - ((float)y / cam.resolution.y) * 2.f;

		// Compute point on plane of focus
		glm::vec3 pFocus = ref + ndcX * H + ndcY * V;

		// Update ray for effect of lens
		segment.ray.origin = cam.position + (cam.up * point.y) + (cam.right * point.x);
		segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
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
	, int geoms_size
	, ShadeableIntersection* intersections
	, int iter
	, Triangle* tris
	, BVHNode* bvhNodes
	, int* trisIndices
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment& pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		Ray ray = pathSegment.ray;
		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];
			ray = pathSegment.ray;
#if MOTION_BLUR
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, pathSegment.remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);
			
			// Jitter ray randomly about a given axis, here faking velocity vector as axis 
			if (glm::length(geom.velocity) > 1.f) {
				ray.origin += u01(rng) * geom.velocity;
			}
#endif
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
			}
			else if ((geom.type == OBJ) || (geom.type = GLTF))
			{
#if BVH
				t = bvhIntersectionTest(geom, bvhNodes, tris, ray, tmp_intersect, tmp_normal);
#elif BB_CULLING
				if (aabbIntersectionTest(geom.aabb, ray, t)) {
					t = meshIntersectionTest(geom, ray, tris, tmp_intersect, tmp_normal);
				}
#else
				t = meshIntersectionTest(geom, ray, tris, tmp_intersect, tmp_normal);
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
			}
		}

		// Not hit
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

__global__ void shadeMaterials(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	if (pathSegments[idx].remainingBounces == 0) return;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	if (intersection.t > 0.f) { // if the intersection exists...
#if DEBUG_BVH
		pathSegments[idx].color = intersection.surfaceNormal * 0.5f + 0.5f;
		return;
#endif
		// Set up the RNG
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.f) {
			pathSegments[idx].color += (materialColor * material.emittance)
				* pathSegments[idx].throughput;
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			//glm::vec3 isect = pathSegments[idx].ray.direction * intersection.t + pathSegments[idx].ray.origin;;
			glm::vec3 isect = getPointOnRay(pathSegments[idx].ray, intersection.t);
			scatterRay(pathSegments[idx], isect, intersection.surfaceNormal, material, rng);
		}
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
	}
	else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0;
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
#if TIMER
	PerformanceTimer timer;
#endif
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

	// Implement motion blur by updating object's tranformation
#if MOTION
		//float timeStep = iter; //wrong, fly out of scene
		// how long we want geom to arrive at the expected position -- here is 500 iteration
	float timeStep = 1 / (hst_scene->state.iterations * 0.1f);

	for (int i = 0; i < hst_scene->geoms.size(); i++)
	{
		hst_scene->geoms[i].translation += hst_scene->geoms[i].velocity * timeStep;
		hst_scene->geoms[i].transform = utilityCore::buildTransformationMatrix(hst_scene->geoms[i].translation, hst_scene->geoms[i].rotation, hst_scene->geoms[i].scale);
		hst_scene->geoms[i].inverseTransform = glm::inverse(hst_scene->geoms[i].transform);
		hst_scene->geoms[i].invTranspose = glm::inverseTranspose(hst_scene->geoms[i].transform);
	}

	cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

#if TIMER
	timer.startGpuTimer();
#endif

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
		// compute first bounce intersections
		if (iter == 1) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, iter
				, dev_tris
				, dev_bvhNodes
				, dev_bvhTriIndices
				);
			checkCUDAError("trace first bounce");
			cudaDeviceSynchronize();

			// cache ray intersections
			if (depth == 0) {
				cudaMemcpy(dev_intersections_first_bounce, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		// for all subsequent iterations, read from cached 
		else {
			if (depth == 0) {
				cudaMemcpy(dev_intersections, dev_intersections_first_bounce, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, iter
					, dev_tris
					, dev_bvhNodes
					, dev_bvhTriIndices
					);
				checkCUDAError("trace non-first bounce");
				cudaDeviceSynchronize();
			}
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, iter
			, dev_tris
			, dev_bvhNodes
			, dev_bvhTriIndices
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MAT
		// shuffle paths to be continugous in memory by material type
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths,
			dev_paths, compare_mat_id());
#endif

		shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth
			);
		checkCUDAError("shadeMaterials failed");

#if COMPACT
		// partition the buffer based on whether the ray path is terminated
		PathSegment* path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_ray_done());
		num_paths = path_end - dev_paths;
		iterationComplete = (num_paths == 0) || (depth == traceDepth);
#else
		iterationComplete = (depth == traceDepth);
#endif

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
		depth++;
	}
#if TIMER
	timer.endGpuTimer();
	std::cout << "with stream compaction: " << timer.getGpuElapsedTimeForPreviousOperation() << "ms" << std::endl;
#endif

	// remember to recover num paths
	num_paths = dev_path_end - dev_paths;
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
