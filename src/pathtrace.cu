#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define SCATTER_ORIGIN_OFFSETMULT 0.001f

#define MAX_ITER 8

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

__device__ inline bool util_math_is_nan(const glm::vec3& v)
{
	return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
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

		if (util_math_is_nan(image[index]))
		{
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else
		{
			// Each thread writes one pixel location in the texture (textel)
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
		pbo[index].w = 0;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
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

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths1);
	cudaFree(dev_paths2);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections1);
	cudaFree(dev_intersections2);
	// TODO: clean up any extra device memory you created

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
__global__ void compute_intersection(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
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
		if (hit_geom_index == -1)//hits nothing
		{
			rayValid[path_index] = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].worldPos = intersect_point;
			rayValid[path_index] = 1;
		}
	}
}

__device__ inline glm::vec2 util_sample_disk_uniform(const glm::vec2& random)
{
	float r = sqrt(random.x);
	float theta = TWO_PI * random.y;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__device__ inline glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random)
{
	glm::vec2 t = util_sample_disk_uniform(random);
	return glm::vec3(t.x, t.y, sqrt(1 - t.x * t.x - t.y * t.y));
}

__device__ inline float util_math_abscos(const glm::vec3& w)
{
	return abs(w.z);
}

__device__ glm::vec3 bxdf_diffuse_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& random, float* pdf, glm::vec3 diffuseAlbedo)
{
	*wi = util_sample_hemisphere_cosine(random);
	*pdf = wo.z > 0 ? util_math_abscos(*wi) * INV_PI : 0;
	return diffuseAlbedo * INV_PI;
}

__device__ inline float util_math_sin_cos_convert(float sinOrCos)
{
	return sqrt(max(1 - sinOrCos * sinOrCos, 0.0f));
}

__device__ float util_transport_frensel_dielectric(float cosThetaI, float etaI, float etaT)
{
	float sinThetaI = util_math_sin_cos_convert(cosThetaI);
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1) return 1;//total reflection
	float cosThetaT = util_math_sin_cos_convert(sinThetaT);
	float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
	return (rparll * rparll + rperpe * rperpe) * 0.5;
}

__device__ inline bool util_geomerty_refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3* wt)
{
	float cosThetaI = glm::dot(wi, n);
	float sin2ThetaI = max(0.0f, 1 - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = sqrt(1 - sin2ThetaT);
	*wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
	return true;
}

__device__ glm::vec3 bxdf_frensel_specular_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& random, float* pdf, glm::vec3 reflectionAlbedo, glm::vec3 refractionAlbedo, glm::vec2 refIdx)
{
	float frensel = util_transport_frensel_dielectric(util_math_abscos(wo), refIdx.x, refIdx.y);
	if (random.x < frensel)
	{
		*wi = glm::vec3(-wo.x, -wo.y, wo.z);
		*pdf = frensel;
		return frensel * reflectionAlbedo / util_math_abscos(*wi);
	}
	else
	{
		glm::vec3 n = glm::dot(wo, glm::vec3(0, 0, 1)) > 0 ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
		glm::vec3 refractedRay;
		if (!util_geomerty_refract(wo, n, refIdx.x / refIdx.y, &refractedRay)) return glm::vec3(0);
		*wi = refractedRay;
		*pdf = 1 - frensel;
		glm::vec3 val = refractionAlbedo * (1 - frensel) * (refIdx.x * refIdx.x) / (refIdx.y, refIdx.y);
		return val / util_math_abscos(*wi);
	}
}

//Get tangent space vectors
__device__ void util_math_get_TBN(const glm::vec3& N, glm::vec3* T, glm::vec3* B)
{
	float x = N.x, y = N.y, z = N.z;
	float sz = z < 0 ? -1 : 1;
	float a = 1.0f / (sz + z);
	float ya = y * a;
	float b = x * ya;
	float c = x * sz;
	(*T) = glm::vec3(c * x * a - 1, sz * b, c);
	(*B) = glm::vec3(b, y * ya - sz, y);
}




__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int* rayValid
	, glm::vec3* image
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.type & MaterialType::emitting) {
			pathSegments[idx].color *= (materialColor * material.emittance);
			rayValid[idx] = 0;
			if (!util_math_is_nan(pathSegments[idx].color))
				image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
		}
		else {
			glm::vec3& woInWorld = pathSegments[idx].ray.direction;
			glm::vec3& N = glm::normalize(intersection.surfaceNormal);
			glm::vec3 B, T;
			util_math_get_TBN(N, &T, &B);
			glm::mat3 TBN(T, B, N);
			glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
			wo = glm::normalize(wo);
			float pdf = 0;
			glm::vec3 wi, bxdf;
			if (material.type & MaterialType::frenselSpecular)
			{
				glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
				bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec3(u01(rng), u01(rng), u01(rng)), &pdf, materialColor, materialColor, iors);
			}
			else
				bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor);
			if (pdf > 0)
			{
				pathSegments[idx].color *= bxdf * util_math_abscos(wi) / pdf;
				glm::vec3 newDir = glm::normalize(TBN * wi);
				glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
				pathSegments[idx].ray.origin = intersection.worldPos + offset * SCATTER_ORIGIN_OFFSETMULT;
				pathSegments[idx].ray.direction = newDir;
				rayValid[idx] = 1;
			}
			else
			{
				rayValid[idx] = 0;
			}
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
		if (!util_math_is_nan(iterationPath.color))
			image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

int compact_rays(int* rayValid,int* rayIndex,int numRays)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths1(dev_paths1), dev_thrust_paths2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections1(dev_intersections1), dev_thrust_intersections2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(dev_thrust_paths1, dev_thrust_paths1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_paths2);
	thrust::scatter_if(dev_thrust_intersections1, dev_thrust_intersections1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersections2);
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return nextNumRays;
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths1);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = dev_path_end - dev_paths1;
	int* rayValid, * rayIndex;
	
	int numRays = num_paths;
	cudaMalloc((void**)&rayValid, sizeof(int) * pixelcount);
	cudaMalloc((void**)&rayIndex, sizeof(int) * pixelcount);
	
	cudaDeviceSynchronize();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (numRays && depth < MAX_ITER) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
		compute_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, numRays
			, dev_paths1
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections1
			, rayValid
			, dev_image
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		numRays = compact_rays(rayValid, rayIndex, numRays);

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
		scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_materials,
			rayValid,
			dev_image
			);

		numRays = compact_rays(rayValid, rayIndex, numRays);

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	if (numRays)
	{
		// Assemble this iteration and apply it to the image
		dim3 numBlocksPixels = (numRays + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> > (numRays, dev_image, dev_paths1);
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	cudaFree(rayValid);
	cudaFree(rayIndex);

	checkCUDAError("pathtrace");
}
