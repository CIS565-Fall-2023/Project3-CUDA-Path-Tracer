#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "depScene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "light.h"
#include "scene.h"
#include "bsdf.h"
#include "bvh.h"
//#include "utilities.cuh"

#define ONE_BOUNCE_DIRECT_LIGHTINIG 1

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

static HostScene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles= nullptr;
static Scene * pa = new Scene("D:\\AndrewChen\\CIS565\\Project3-CUDA-Path-Tracer\\scenes\\pathtracer_test.glb");
static BSDFStruct * dev_bsdfStructs = nullptr;
static BVHAccel * bvh = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

__global__ void initAreaLightFromObject(HostScene * scene, Light** lights, Geom* geoms, Material * mats, int geoms_size) {
	//lights[0] = new AreaLight(glm::vec3(1.0f, 1.0f, 1.0f), 5.0f);
	int light_index = 0;
	for (size_t i = 0; i < geoms_size; i++)
	{
		auto geom = geoms[i];
		auto mat = mats[geom.materialid];
		if (mat.emittance > EPSILON) {
			lights[light_index] = new AreaLight(mat.color, mat.emittance);
			light_index++;
		}
	}
}

void pathtraceInit(HostScene* scene) {
	
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	bvh = new BVHAccel();
	bvh->buildBVH(pa->triangles);
	bvh->traverseBVHNonSerialized();
	system("pause");
	auto triangles = pa->triangles.data();
	cudaMalloc(&dev_triangles, pa->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, triangles, pa->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);


	auto bsdfStructs = pa->bsdfStructs.data();
	cudaMalloc(&dev_bsdfStructs, pa->bsdfStructs.size() * sizeof(BSDFStruct));
	cudaMemcpy(dev_bsdfStructs, bsdfStructs, pa->bsdfStructs.size() * sizeof(BSDFStruct), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
	cudaFree(dev_bsdfStructs);
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
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.constantTerm = glm::vec3(1.0f, 1.0f, 1.0f);
	}
}

//TODO: Change to BVH in future!
__device__ bool intersectCore(Triangle * triangles, int triangles_size, Ray & ray, ShadeableIntersection& intersection) {

	for (int i = 0; i < triangles_size; i++)
	{
		triangles[i].intersect(ray, &intersection);
	}
	return intersection.t > EPSILON;
}

__global__ void intersect(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Triangle* triangles
	, int triangles_size
	, ShadeableIntersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		Ray & ray = pathSegments[path_index].ray;
		ray.min_t = 0.0;
		ray.max_t = FLT_MAX;
		intersections[path_index].t = -1;
		intersectCore(triangles, triangles_size, pathSegments[path_index].ray, intersections[path_index]);
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
__global__ void shadeBSDF(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, BSDFStruct * bsdfStructs
	, Triangle * triangles
	, int triangles_size
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			BSDFStruct & bsdfStruct = bsdfStructs[intersection.materialId];
			// If the material indicates that the object was a light, "light" the ray
			if (bsdfStruct.bsdfType == BSDFType::EMISSIVE) {
				pathSegments[idx].remainingBounces = 0;
				pathSegment.color += (bsdfStruct.emissiveFactor * bsdfStruct.strength * pathSegment.constantTerm);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				glm::mat3x3 o2w;
				make_coord_space(o2w, intersection.surfaceNormal);
				glm::mat3x3 w2o(glm::transpose(o2w));
				glm::vec3 intersect = pathSegment.ray.at(intersection.t);
				glm::vec3 bsdf = bsdfStruct.reflectance * INV_PI; // Hard coded for now
				glm::vec3 L_direct;
				float pdf;
				glm::vec3 wo = -pathSegment.ray.direction;
				glm::vec3 wi;
#if  ONE_BOUNCE_DIRECT_LIGHTINIG
				/* Estimate one bounce direct lighting */
				int n_sample = 10;
				int n_valid_sample = 0;
				for (size_t i = 0; i < n_sample; i++)
				{
					sample_f(bsdfStruct, wo, wi, &pdf, rng);
					float cosineTerm = abs(wi.z);
					Ray one_bounce_ray;
					one_bounce_ray.direction = o2w * wi;
					one_bounce_ray.origin = intersect;
					one_bounce_ray.min_t = EPSILON;
					one_bounce_ray.max_t = FLT_MAX;
					ShadeableIntersection oneBounceIntersection;
					oneBounceIntersection.t = -1.0f;

					if (intersectCore(triangles, triangles_size, one_bounce_ray, oneBounceIntersection)){
						auto oneBounceBSDF = bsdfStructs[oneBounceIntersection.materialId];
						if (oneBounceBSDF.bsdfType == BSDFType::EMISSIVE) {
							n_valid_sample++;
							auto Li = oneBounceBSDF.emissiveFactor * oneBounceBSDF.strength;

							// TODO: Figure out why is it still not so similar to the reference rendered image?
							//		May be checkout the light propogation chapter of pbrt to find out how we use direct light estimation?
							L_direct += Li * bsdf * cosineTerm;
						}
					}
				}

				float one_over_n = 1.0f / (n_valid_sample + 1);
				pathSegment.color += (L_direct * pathSegment.constantTerm * one_over_n);
#endif //  ONE_BOUNCE_DIRECT_LIGHTINIG
				sample_f(bsdfStruct, wo, wi, &pdf, rng);
				float cosineTerm = abs(wi.z);
#if ONE_BOUNCE_DIRECT_LIGHTINIG
				pathSegment.constantTerm *= (bsdf * cosineTerm * one_over_n / pdf);
#else
				pathSegment.constantTerm *= (bsdf * cosineTerm / pdf);
#endif			
				pathSegment.ray.origin = intersect;
				pathSegment.ray.direction = o2w * wi;
				pathSegment.remainingBounces--;

				//scatterRay(pathSegments[idx], pathSegments[idx].ray.at(intersection.t), intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].remainingBounces = 0;
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

struct HasHit{
    __host__ __device__ bool operator()(const PathSegment & path) const {
        return path.remainingBounces != 0;
    }
};

struct NoHit {
	__host__ __device__ bool operator()(const PathSegment& path) const {
		return path.remainingBounces == 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	checkCUDAError("before generate camera ray");

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

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		int primitiveSize = pa->getPrimitiveSize();
		checkCUDAError("getPrimitiveSize");

		intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_triangles
			, pa->triangles.size()
			, dev_intersections
			);

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_bsdfStructs,
			dev_triangles,
			pa->triangles.size()
			);

		checkCUDAError("shadeMaterial failed\n");
		// Use dev_paths for compaction result
		// check if compacted dev_intersections has zero element, if it is, then iteraionComplete should be true.
		// If no, then we might be able to store the pixel that has not finished dev_intersection to largely decrease the number 
		// of pixels that are required to continue raytracing.
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, HasHit());
		num_paths = dev_path_end - dev_paths;

		iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.
		// iterationComplete = (true);
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	// dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
