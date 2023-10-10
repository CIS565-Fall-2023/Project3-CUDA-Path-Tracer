#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

// toggle material sorting
#define CONTIGUOUS_MATERIAL 1

// toggle first bounce intersections
// not available when CACHE is ON or FISHEYE is ON
#define CACHE_FIRST_BOUNCE 0

// Depth of field
#define CAM_APERTURE_DIAM 0.0f // set to 0.0f for no depth of field
#define CAM_FOCAL_DIST 11.58f

// toggle fisheye camera
// not available when CACHE is ON
// 0 for pin-hole, 1 for fish-eye, 2 for panorama
#define CAM_STYLE 0
#define CAM_FISHEYE_MAX_ANGLE PI/2.0f

// motion blur: set to 0 to disable
#define CAM_SHUTTER_TIME 0

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
#if CONTIGUOUS_MATERIAL == 1
static int* dev_indices = NULL; // for materialId grouping
static PathSegment* dev_paths_tmp = NULL; // Ping-Pong buffer for reordering
static ShadeableIntersection* dev_intersections_tmp = NULL;
#endif // CONTIGUOUS_MATERIAL
#if CACHE_FIRST_BOUNCE == 1
static PathSegment* dev_first_paths = NULL;
static ShadeableIntersection* dev_first_intersections = NULL;
#endif // CACHE_FIRST_BOUNCE
#if CAM_SHUTTER_TIME != 0
// buffer to store moved geoms
static Geom* dev_geoms_tmp = NULL;
#endif

// identifier rule for thrust::remove_if
struct is_path_terminated {
	__host__ __device__ bool operator()(const PathSegment& segment) {
		return segment.remainingBounces <= 0;
	}
};

#if CONTIGUOUS_MATERIAL == 1
// comparator to sort pathSegments and shadeableIntersections based on materialId and t
struct IntersectionComparator {
	ShadeableIntersection* intersections;
	IntersectionComparator(ShadeableIntersection* _intersections)
		: intersections(_intersections) {}
	
	__device__	bool operator()(int i, int j) const {
		return intersections[i].materialId < intersections[j].materialId;
	}
};
#endif // CONTIGUOUS_MATERIAL


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
#if CONTIGUOUS_MATERIAL == 1
	cudaMalloc(&dev_indices, pixelcount * sizeof(int));
	cudaMalloc(&dev_paths_tmp, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_intersections_tmp, pixelcount * sizeof(ShadeableIntersection));
#endif // CONTIGUOUS_MATERIAL

#if CACHE_FIRST_BOUNCE == 1
	cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	const int traceDepth = hst_scene->state.traceDepth;
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	// 1D block for path tracing
	const int blockSize1d = 128;
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, 0, traceDepth, dev_first_paths);
	checkCUDAError("generate initial camera ray");
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	// trace first bounce
	dim3 numblocksPathSegmentTracing = (pixelcount + blockSize1d - 1) / blockSize1d;
	computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
		0,
		pixelcount,
		dev_first_paths,
		dev_geoms,
		hst_scene->geoms.size(),
		dev_first_intersections
	);
	checkCUDAError("trace first bounce");
	cudaDeviceSynchronize();
#endif // CACHE_FIRST_BOUNCE
#if CAM_SHUTTER_TIME != 0
	cudaMalloc(&dev_geoms_tmp, scene->geoms.size() * sizeof(Geom));
#endif
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if CONTIGUOUS_MATERIAL == 1
	cudaFree(dev_indices);
	cudaFree(dev_paths_tmp);
	cudaFree(dev_intersections_tmp);
#endif // CONTIGUOUS_MATERIAL
#if CACHE_FIRST_BOUNCE == 1
	cudaFree(dev_first_paths);
	cudaFree(dev_first_intersections);
#endif // CACHE_FIRST_BOUNCE
#if CAM_SHUTTER_TIME != 0
	cudaFree(dev_geoms_tmp);
#endif
	checkCUDAError("pathtraceFree");
}

#if CAM_STYLE == 1
// calculate fish eye coordinate, given pixel coordinates in range [-1, 1]
__host__ __device__ glm::vec3 calcFishEye(float u, float v) {
	float radiant_dist = sqrtf(u * u + v * v);
	if (radiant_dist > 1.0f) {
		return glm::vec3(0.f);
	}
	// compute angles
	float theta = radiant_dist * CAM_FISHEYE_MAX_ANGLE;
	float phi = atan2f(v, u);
	return glm::vec3(
		sinf(theta) * cosf(phi),
		sinf(theta) * sinf(phi),
		cosf(theta)
	);
}
#elif CAM_STYLE == 2
__host__ __device__ glm::vec3 calcPanorama(float u, float v) {
	// compute cylindar angles
	float elevation = 0.5f * PI * (v);
	float azimuth = ((u)) * PI;
	// convert into local cartesian coordinates
	return glm::vec3(
		cosf(elevation) * sinf(azimuth),
		sinf(elevation),
		cosf(elevation) * cosf(azimuth)
	);
}
#endif // CAM_STYLE


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

#if CACHE_FIRST_BOUNCE == 1
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#else
		// jittering rays, subpixel for antialiasing
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> uniform_pixel(-0.5f, 0.5f); // can shift at most 0.5 pixel
		float jit_x_num_pixel = uniform_pixel(rng);
		float jit_y_num_pixel = uniform_pixel(rng);

		// jittering within aperture for depth-of-field
		thrust::uniform_real_distribution<float> uniform_radius(0, 0.5f * CAM_APERTURE_DIAM);
		thrust::uniform_real_distribution<float> uniform_angle(0, 2.0f * PI);
		float radius = uniform_radius(rng);
		float angle = uniform_angle(rng);
		float jit_aperture_x = radius * cosf(angle);
		float jit_aperture_y = radius * sinf(angle);

#if CAM_STYLE != 0
		// some cam type specified, have to compute pixel coordinates [-1, 1]
		float u = -((float)x - (float)cam.resolution.x * 0.5f) / ((float)cam.resolution.x * 0.5f);
		float v = -((float)y - (float)cam.resolution.y * 0.5f) / ((float)cam.resolution.y * 0.5f);
#endif
#if CAM_STYLE == 1
		// FishEye camera
		glm::vec3 fisheye_dir = calcFishEye(u, v);
		if (glm::length(fisheye_dir) < 0.001f) {
			segment.color = glm::vec3(0.f);
			segment.remainingBounces = 0;
			return;
		}
		segment.ray.direction = glm::normalize(cam.view * fisheye_dir.z +
																					 cam.right * fisheye_dir.x +
																					 cam.up * fisheye_dir.y);
#elif CAM_STYLE == 2
		// Panorama
		glm::vec3 pano_dir = calcPanorama(u, v);
		segment.ray.direction = glm::normalize(cam.view * pano_dir.z +
                                           cam.right * pano_dir.x +
                                           cam.up * pano_dir.y);
#else
		// setting origin (randomized on aperture)
		segment.ray.origin = cam.position + cam.right * jit_aperture_x + cam.up * jit_aperture_y;

		glm::vec3 focal_point = cam.position + glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jit_x_num_pixel - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jit_y_num_pixel - (float)cam.resolution.y * 0.5f)) * CAM_FOCAL_DIST;
		segment.ray.direction = glm::normalize(focal_point - segment.ray.origin);
#endif
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
		bool tmp_outside = true;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
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
				outside = tmp_outside;
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
			intersections[path_index].outside = outside;
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

// the true shader
__global__ void shadeMaterial(
	int depth, // must be passed in for random seeding
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) {
		// out of bound
		return;
	}

	ShadeableIntersection intersection = shadeableIntersections[idx];

	if (intersection.t <= 0.0f) {
		// no intersection color as black
		pathSegments[idx].color = glm::vec3(0.0f);
		// mark this path as terminated
		pathSegments[idx].remainingBounces = 0;
		return;
	}

	// there is a valid intersection
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	Material material = materials[intersection.materialId];
	Ray ray_in = pathSegments[idx].ray;
	glm::vec3 intersection_point = ray_in.origin + ray_in.direction * intersection.t;
	scatterRay(pathSegments[idx], intersection_point, intersection.surfaceNormal, intersection.outside, material, rng);
}

// Add output of terminated pathSegments to the overall image
__global__ void partialGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nPaths) {
		return;
	}
	PathSegment iterationPath = iterationPaths[index];
	if (iterationPath.remainingBounces <= 0) {
		image[iterationPath.pixelIndex] += iterationPath.color;
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

#if CAM_SHUTTER_TIME != 0
// a GPU copy from CoreUtilities
__host__ __device__ glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}

// update matrices in g
__host__ __device__ void updateGeom(Geom& g) {
	g.transform = buildTransformationMatrix(
		g.translation, g.rotation, g.scale);
	g.inverseTransform = glm::inverse(g.transform);
	g.invTranspose = glm::inverseTranspose(g.transform);
}

// compute geoms at a specific time
__global__ void calcGeoms(Geom* geoms, Geom* geoms_tmp, int num_geoms, float time) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= num_geoms) {
		return;
	}
	Geom g = geoms[index];
	if (glm::length(g.velocity) < 0.001f) {
		geoms_tmp[index] = g;
		return;
	}
	g.translation += g.velocity * time;
	updateGeom(g);
	geoms_tmp[index] = g;
	return;
}
#endif

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
#if CAM_SHUTTER_TIME != 0
	// randomize a time between 0 and shutter time
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, 0, 0);
	thrust::uniform_real_distribution<float> u01(0, CAM_SHUTTER_TIME);
	float time = u01(rng);
	time = powf(time, 0.4f);

	// generate new geoms based on time
	int fullBlocksPerGeom = (hst_scene->geoms.size() + blockSize1d - 1) / blockSize1d;
	calcGeoms<<<fullBlocksPerGeom, blockSize1d>>>(dev_geoms, dev_geoms_tmp, hst_scene->geoms.size(), time);
#endif

#if CACHE_FIRST_BOUNCE == 1
	cudaMemcpy(dev_paths, dev_first_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
#else
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif // CACHE_FIRST_BOUNCE

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// tracing
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_FIRST_BOUNCE == 1
		if (depth == 0) {
			// first bounce, copy intersections instead of compute them
			cudaMemcpy(dev_intersections, dev_first_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_intersections
			);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth,
			num_paths,
			dev_paths,
#if CAM_SHUTTER_TIME != 0
			dev_geoms_tmp,
#else
			dev_geoms,
#endif
			hst_scene->geoms.size(),
			dev_intersections
		);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif // CACHE_FIRST_BOUNCE
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if CONTIGUOUS_MATERIAL == 1
		// sort PathSegments and ShadeableIntersections
		// locate range in dev_indices to initialize the sequence
		thrust::device_ptr<int> dev_thrust_indices(dev_indices);
		thrust::device_ptr<int> dev_thrust_indices_end(dev_indices + num_paths);
		// init sequence in this range
		thrust::sequence(dev_thrust_indices, dev_thrust_indices_end);
		IntersectionComparator comp(dev_intersections);
		thrust::sort(dev_thrust_indices, dev_thrust_indices_end, comp);
		// gather sorted items in tmp arrays
		thrust::gather(dev_thrust_indices, dev_thrust_indices_end, thrust::device_pointer_cast(dev_paths), thrust::device_pointer_cast(dev_paths_tmp));
		thrust::gather(dev_thrust_indices, dev_thrust_indices_end, thrust::device_pointer_cast(dev_intersections), thrust::device_pointer_cast(dev_intersections_tmp));
		// swap ping-pong
		std::swap(dev_paths, dev_paths_tmp);
		std::swap(dev_intersections, dev_intersections_tmp);
#endif // CONTIGUOUS_MATERIAL

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth,
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);

		// gather outputs of paths that are terminated
		partialGather<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_image, dev_paths);

		// stream compaction using thrust::remove_if
		thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
		thrust::device_ptr<PathSegment> dev_thrust_paths_end = thrust::remove_if(
			thrust::device,
			dev_thrust_paths,
			dev_thrust_paths + num_paths,
			is_path_terminated()
		);

		// update remaining path count
		num_paths = dev_thrust_paths_end - dev_thrust_paths;

		iterationComplete = num_paths == 0; // TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	//dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
