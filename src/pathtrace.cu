#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
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
#include <OpenImageDenoise/oidn.hpp>


#define ERRORCHECK 1

#define SORT_MATERIALS 0

#define CACHE_FIRST_BOUNCE 1

// enable 4x Stochastic Sample Anti-Aliasing
#define ANTIALIASING 0

#define DIRECT_LIGHT 0

#define IMAGE_DENOISE 1

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
static glm::vec3* dev_final_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle *dev_meshes = NULL;
#if IMAGE_DENOISE
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
#endif 

#if DIRECT_LIGHT
static int * dev_light_indices = NULL;
static int num_lights = 0;
#endif


#ifdef CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_first_bounce_intersections = NULL;
static PathSegment* dev_first_bounce_paths = NULL;
#endif

// TODO: static variables for device memory, any extra info you need, etc
// ...

#if IMAGE_DENOISE
void imageDenoise() {
	// Create an Intel(R) Open Image Denoise device
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	// check which device is being used
	int d = oidnGetNumPhysicalDevices();

	// check the devices avaiable, range from 0 to d-1, rank by performance
	//for (int i = 0; i < d; i++) {
	//	const char* para = "name";
	//	const char* name = oidnGetPhysicalDeviceString(i, para);
	//	cout << "device " << i << " type: " << name << endl;
	//}

	cout << "image denoise start" << endl;
	int width = hst_scene->state.camera.resolution.x;
	int height = hst_scene->state.camera.resolution.y;
	// Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
	//oidn::BufferRef colorBuf = device.newBuffer(width * height * 3 * sizeof(float));
	//oidn::BufferRef albedoBuf = ...

	// Create a filter for denoising a beauty (color) image using optional auxiliary images too
	// This can be an expensive operation, so try no to create a new filter for every image!
	oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
	filter.setImage("color", dev_image, oidn::Format::Float3, width, height); // beauty
	filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height); // auxiliary
	filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height); // auxiliary
	filter.setImage("output", dev_denoised_image, oidn::Format::Float3, width, height); // denoised beauty
	filter.set("hdr", true); // beauty image is HDR
	//filter.set("cleanAux", true); // auxiliary images will be prefiltered, make sure contains as less noise as possible
	filter.commit();

	// Fill the input image buffers
	//float* colorPtr = (float*)colorBuf.getData();

	// Create a separate filter for denoising an auxiliary albedo image (in-place)
	oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
	albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
	albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
	albedoFilter.commit();

	//// Create a separate filter for denoising an auxiliary normal image (in-place)
	oidn::FilterRef normalFilter = device.newFilter("RT"); // same filter type as for beauty
	normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
	normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
	normalFilter.commit();

	// Prefilter the auxiliary images
	albedoFilter.execute();
	normalFilter.execute();

	// Filter the beauty image
	filter.execute();

	// Check for errors
	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
		std::cout << "Error: " << errorMessage << std::endl;

}


__global__ 
void setPrefilter(
	ShadeableIntersection* shadeableIntersections, int num_path,
	PathSegment* pathSegments, glm::vec3* albedo, glm::vec3* normal) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < num_path) {
		ShadeableIntersection intersection = shadeableIntersections[index];
		PathSegment& pathSegment = pathSegments[index];

		normal[pathSegment.pixelIndex] = intersection.surfaceNormal;
		albedo[pathSegment.pixelIndex] = pathSegment.color;
	}
}

#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
#if ANTIALIASING

	const int pixelcount = 4 * cam.resolution.x * cam.resolution.y;
	cudaMalloc(&dev_image, pixelcount  * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount  * sizeof(glm::vec3));

	cudaMalloc(&dev_final_image, pixelcount / 4 * sizeof(glm::vec3));
	cudaMemset(dev_final_image, 0, pixelcount / 4 * sizeof(glm::vec3));
#else

	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
#endif

	
	//cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	//cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if IMAGE_DENOISE
	cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
#endif // OIDN

	// TODO: initialize any extra device memeory you need
#if DIRECT_LIGHT
	num_lights = 0;
	std::vector<int> light_indices;
	for (int i = 0; i < scene->geoms.size(); i++) {
		Geom &geom = scene->geoms[i];
		if (scene->materials[geom.materialid].emittance > 0) {
			// find a light source
			light_indices.push_back(i);
			num_lights++;
		}
	}
	cudaMalloc(&dev_light_indices, num_lights * sizeof(int));
	cudaMemcpy(dev_light_indices, light_indices.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
#endif


#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));
#endif
	cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Triangle));
	cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
	cout << "pathtraceInit finished" << endl;
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_final_image);
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_meshes);
#if IMAGE_DENOISE
	cudaFree(dev_denoised_image);
	cudaFree(dev_albedo);
	cudaFree(dev_normal);
#endif

#if DIRECT_LIGHT
	cudaFree(dev_light_indices);
#endif
	// TODO: clean up any extra device memory you created
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_first_bounce_intersections);
	cudaFree(dev_first_bounce_paths);
#endif

	checkCUDAError("pathtraceFree");
}

__host__ __device__ void concentricSampleDisk(float* sampledX, float* sampledY, thrust::default_random_engine& rng) {
	// Generate two random numbers between -1 and 1
	thrust::uniform_real_distribution<float> uniformDistribution(-1, 1);
	float randomX = uniformDistribution(rng);
	float randomY = uniformDistribution(rng);

	// Map random points to concentric disk
	float radius, theta;
	if (randomX * randomX > randomY * randomY) {
		radius = randomX;
		theta = (PI / 4.f) * (randomY / randomX);
	}
	else {
		radius = randomY;
		theta = (PI / 2.f) - (PI / 4.f) * (randomX / randomY);
	}

	// Convert polar coordinates to Cartesian coordinates
	*sampledX = radius * cosf(theta);
	*sampledY = radius * sinf(theta);
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

#if ANTIALIASING
	if (x < 2 * cam.resolution.x && y < 2 * cam.resolution.y) {
		int index = x + (y * cam.resolution.x * 2);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		// Initialize random number generator
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		
		// Initialize uniform distribution in two dimensions
		thrust::uniform_real_distribution<float> u01(-1, 1);
		thrust::uniform_real_distribution<float> u02(-1, 1);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) * 4.f - (float)cam.resolution.x * 2 * 0.5f) / 2.f
			- cam.up * cam.pixelLength.y * ((float)y + u02(rng) * 4.f - (float)cam.resolution.y * 2 * 0.5f) / 2.f
		);

		if (cam.lensRadius > 0) {
			// Initialize random number generator
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			//thrust::uniform_real_distribution<float> uniformDistribution(-1, 1);

			// Sample a point on the lens using the concentricSampleDisk function
			float lensSampleX, lensSampleY;
			concentricSampleDisk(&lensSampleX, &lensSampleY, rng);

			// Scale the sample by the lens radius
			glm::vec3 lensPosition = cam.lensRadius * glm::vec3(lensSampleX, lensSampleY, 0.f);

			// Calculate the distance from the lens to the focal plane
			float focalPlaneIntersectionDistance = cam.focalLength / -segment.ray.direction.z;

			// Determine the point of intersection on the focal plane
			glm::vec3 focalPlaneIntersection = focalPlaneIntersectionDistance * segment.ray.direction;

			// Update ray origin and direction for the effect of the lens
			segment.ray.origin += lensPosition;
			segment.ray.direction = glm::normalize(focalPlaneIntersection - lensPosition);
		}

		segment.ray.isShadow = false;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
#else
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

		if (cam.lensRadius > 0) {
			// Initialize random number generator
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			//thrust::uniform_real_distribution<float> uniformDistribution(-1, 1);

			// Sample a point on the lens using the concentricSampleDisk function
			float lensSampleX, lensSampleY;
			concentricSampleDisk(&lensSampleX, &lensSampleY, rng);

			// Scale the sample by the lens radius
			glm::vec3 lensPosition = cam.lensRadius * glm::vec3(lensSampleX, lensSampleY, 0.f);

			// Calculate the distance from the lens to the focal plane
			float focalPlaneIntersectionDistance = cam.focalLength / -segment.ray.direction.z;

			// Determine the point of intersection on the focal plane
			glm::vec3 focalPlaneIntersection = focalPlaneIntersectionDistance * segment.ray.direction;

			// Update ray origin and direction for the effect of the lens
			segment.ray.origin += lensPosition;
			segment.ray.direction = glm::normalize(focalPlaneIntersection - lensPosition);
		}

		segment.ray.isShadow = false;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
#endif
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
	, Triangle* meshes
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
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, meshes, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
		}
	}
}

struct material_sort
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
	{
		return a.materialId < b.materialId;
	}
};

struct is_path_terminated
{
	__host__ __device__
		bool operator()(const PathSegment& seg)
	{
		return seg.remainingBounces > 0;
	}
};


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
				pathSegments[idx].remainingBounces = 0;

				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not
				scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			// ray flew into the sky
			pathSegments[idx].remainingBounces = 0;
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeWithDirectLight(
	int iter, int num_paths, ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments, Material* materials, 
	int* lightindices, int light_num, Geom* geoms)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) {
		return;
	}

	ShadeableIntersection intersection = shadeableIntersections[idx];
	PathSegment& pathSegment = pathSegments[idx];

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	Material material = materials[intersection.materialId];
	glm::vec3 intersection_pos = getPointOnRay(pathSegment.ray, intersection.t);

	if (intersection.t <= 0.0f) {
		// No intersection: terminate and color the ray black
		pathSegment.remainingBounces = 0;
		pathSegment.color = glm::vec3(0.0f);
		return;
	}

	if (material.emittance > 0.0f) {
		// Light source: terminate the ray and adjust its color
		pathSegment.remainingBounces = 0;
		pathSegment.color *= (material.color * material.emittance);
	}
	else if (pathSegment.remainingBounces == 1) {
		if (pathSegment.ray.isShadow) {
			// If it's a shadow ray that doesn't reach lightsource
			pathSegment.remainingBounces = 0;
			pathSegment.color = glm::vec3(0.f);
		}
		else {
			// Sample a point from a random light 
			Geom& lightsource = geoms[lightindices[int(u01(rng) * light_num)]];
			glm::vec3 randomLightPoint = multiplyMV(lightsource.transform, glm::vec4(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f, 1.f));

			pathSegment.ray.origin = intersection_pos;
			pathSegment.ray.direction = glm::normalize(randomLightPoint - intersection_pos);
			pathSegment.ray.isShadow = true;
			
			pathSegment.color *= (material.color * abs(glm::dot(pathSegment.ray.direction, intersection.surfaceNormal)) / 2.f);
			
		}
	}
	else {
		scatterRay(pathSegment, intersection_pos, intersection.surfaceNormal, material, rng);
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

// averageWeight the 4X images into one image
__global__ void averageWeightImages(glm::ivec2 resolution, glm::vec3* inputImage, glm::vec3* outputImage)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int out_index = x + (y * resolution.x);
		int in_index = 2 * x + (2 * y * resolution.x * 2);
		outputImage[out_index] = (inputImage[in_index] + inputImage[in_index + 1] + inputImage[in_index + resolution.x * 2] + inputImage[in_index + resolution.x * 2 + 1]) / 4.f;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;

#if ANTIALIASING
	const int pixelcount = 4 * cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x * 2 + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y * 2 + blockSize2d.y - 1) / blockSize2d.y);

	const dim3 finalblocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
#else

	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

#endif
	// 1D block for path tracing
	const int blockSize1d = 128;

	//cout << "iter: " << iter << endl;

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
#if CACHE_FIRST_BOUNCE
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_first_bounce_paths);
		checkCUDAError("generate camera ray");
 		cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else {
		cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}

#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_FIRST_BOUNCE
		if (depth == 0 && iter > 1) {
			// copy the first bounce intersections for later iterations
			cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0 && iter == 1) {
			cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
			// tracing

			//cout << "doing first bounce" << endl;
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_meshes
				, dev_first_bounce_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			//cout << "done first bounce" << endl;

			

			// cache the first bounce intersections
			//cudaMemcpy(dev_first_bounce_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			// tracing

			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_meshes
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_meshes
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

#if SORT_MATERIALS
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_sort());
#endif


#if DIRECT_LIGHT
		shadeWithDirectLight << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_light_indices,
			num_lights,
			dev_geoms
			);
#else
		//cout << "shading" << endl;
		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		//cout << "done shading" << endl;
#endif

#if IMAGE_DENOISE
		if (depth == 1) {
			setPrefilter << <numblocksPathSegmentTracing, blockSize1d >> > (
				dev_intersections,
				num_paths,
				dev_paths,
				dev_albedo,
				dev_normal
				);
			checkCUDAError("setPrefilter");
		}
#endif

		/*iterationComplete = depth == 5;*/
		//cout << "old num_paths: " << num_paths << endl;
		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, is_path_terminated());
		num_paths = dev_path_end - dev_paths;
		//cout << "new num_paths: " << num_paths << endl;

		if (num_paths == 0) {
			//cout << "num_paths == 0" << endl;
			iterationComplete = true; // TODO: should be based off stream compaction results.
		};

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

	//cout << "starting finalGather" << endl;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);
	//cout << "ending finalGather" << endl;
	///////////////////////////////////////////////////////////////////////////


#if ANTIALIASING

	averageWeightImages << <finalblocksPerGrid2d, blockSize2d >> > (cam.resolution, dev_image, dev_final_image);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_final_image);
	//cout << "sendImageToPBO finished" << endl;
	// Retrieve image from GPU

#else

#if IMAGE_DENOISE
		imageDenoise();
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image);
#else

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	//cout << "sendImageToPBO finished" << endl;
	// Retrieve image from GPU
#endif

#endif

#if ANTIALIASING
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount / 4 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif
	checkCUDAError("pathtrace");
}
