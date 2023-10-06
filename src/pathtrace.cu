#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <curand_kernel.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define BlockSize 128
#define STREAM_COMPACTION 1
#define SORT_MATERIAL 1
#define CACHE_FIRST_BOUNCE 1
#define ANTI_ALIASING 1
#define DIRECT_LIGHT 0
#define MAX_OCTREE_DEPTH 4
#define MAX_POTENTIAL_INTERSECTIONS 16
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
// ...
static ShadeableIntersection* dev_cache_intersections = NULL;
static Triangle* dev_triangles = NULL;
static glm::vec4* dev_textures = NULL;
static Geom* lights = NULL;
//static OctreeNode* dev_octree = NULL;  // Global variable


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
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(glm::vec4));
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

	cudaMalloc(&lights, scene->lightNum * sizeof(Geom));
	cudaMemcpy(lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	// 1. Determine the bounding box of the entire scene
	glm::vec3 globalMinCorner = dev_geoms[0].translation;
	glm::vec3 globalMaxCorner = dev_geoms[0].translation;
	for (int i = 0; i < scene->geoms.size(); i++) {
		glm::vec3 geomPosition = dev_geoms[i].translation;
		globalMinCorner = glm::min(globalMinCorner, geomPosition);
		globalMaxCorner = glm::max(globalMaxCorner, geomPosition);
	}

	// 2. Build the octree on the CPU
	//OctreeNode* root = buildOctree(scene->geoms, globalMinCorner, globalMaxCorner, MAX_OCTREE_DEPTH);

	// 3. Allocate memory on the GPU and copy the octree to it
	/*cudaMalloc(&dev_octree, sizeof(OctreeNode));
	cudaMemcpy(dev_octree, root, sizeof(OctreeNode), cudaMemcpyHostToDevice);*/

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cache_intersections);
	cudaFree(dev_triangles);
	cudaFree(dev_textures);
	cudaFree(lights);
	//cudaFree(dev_octree);
	checkCUDAError("pathtraceFree");
}

#define M_PI 3.14159265358979323846
__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	if (uOffset.x == 0.f && uOffset.y == 0.f)
		return glm::vec2(0.f);

	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = (M_PI / 4.f) * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = (M_PI / 2.f) - (M_PI / 4.f) * (uOffset.x / uOffset.y);
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
		// Antialiasing by jittering the ray
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		float randomX = u01(rng);
		float randomY = u01(rng);
		//Stochastic Sampled Antialiasing
#if     ANTI_ALIASING
		float jitterAmount = 0.5f;
		float jitterX = (randomX - 0.5f) * jitterAmount;
		float jitterY = (randomY - 0.5f) * jitterAmount;
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

		// Depth of Field effect
		// Ref: PBTR 6.2
		if (cam.aperture > 0.0f) {
			float lensRadius = cam.aperture / 2.0f;
			glm::vec2 randomCircle = ConcentricSampleDisk(glm::vec2(randomX, randomY));
			glm::vec2 pointOnLens = lensRadius * randomCircle;

			float focalDistance = length(cam.lookAt - cam.position) * lensRadius / cam.aperture;
			glm::vec3 focalPoint = segment.ray.origin + focalDistance * segment.ray.direction;

			// Update ray's origin and direction for Depth of Field
			segment.ray.origin += cam.right * pointOnLens.x + cam.up * pointOnLens.y;
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__host__ __device__ bool checkMeshhBoundingBox(Geom& geom, Ray& ray) {
	Ray q;
	q.origin = glm::vec3(geom.inverseTransform * glm::vec4(ray.origin, 1.0f));
	q.direction = glm::normalize(glm::vec3(geom.inverseTransform * glm::vec4(ray.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;

	for (int xyz = 0; xyz < 3; ++xyz) {
		if (glm::abs(q.direction[xyz]) > 0.00001f) {
			float t1 = (geom.boundingBoxMin[xyz] - q.origin[xyz]) / q.direction[xyz];
			float t2 = (geom.boundingBoxMax[xyz] - q.origin[xyz]) / q.direction[xyz];
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));
		}
	}

	return tmax >= tmin && tmax > 0;
}

__host__ __device__ bool intersectBoundingBox(const glm::vec3& minCorner, const glm::vec3& maxCorner, const Ray& ray) {
	float tmin = (minCorner.x - ray.origin.x) / ray.direction.x;
	float tmax = (maxCorner.x - ray.origin.x) / ray.direction.x;

	if (tmin > tmax) std::swap(tmin, tmax);

	float tymin = (minCorner.y - ray.origin.y) / ray.direction.y;
	float tymax = (maxCorner.y - ray.origin.y) / ray.direction.y;

	if (tymin > tymax) std::swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (minCorner.z - ray.origin.z) / ray.direction.z;
	float tzmax = (maxCorner.z - ray.origin.z) / ray.direction.z;

	if (tzmin > tzmax) std::swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin < 0 && tmax < 0) // both intersections are behind the ray's origin
		return false;

	return true;
}


__host__ __device__ int intersectOctree(const OctreeNode* node, const Ray& ray, Triangle* triangles, Geom* potentialIntersections, int& count) {
	if (!node || !intersectBoundingBox(node->minCorner, node->maxCorner, ray)) {
		return count;
	}

	for (const Geom& obj : node->objects) {
		bool intersects = false;
		glm::vec3 tmp_intersect, tmp_normal;
		glm::vec2 tmp_uv;
		bool outside = true;  // Assuming you have this from your previous context

		switch (obj.type) {
		case CUBE:
			intersects = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside) > 0.0f;
			break;
		case SPHERE:
			intersects = sphereIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside) > 0.0f;
			break;
		case OBJMESH:
			intersects = triangleIntersectionTest(obj, ray, tmp_intersect, triangles, obj.triangleIdStart, obj.triangleIdEnd, tmp_normal, outside, tmp_uv) > 0.0f;
			break;
		default:
			break;
		}

		if (intersects) {
			potentialIntersections[count] = obj;
			count++;
		}
	}

	for (int i = 0; i < 8; ++i) {
		count = intersectOctree(node->children[i], ray, triangles, potentialIntersections, count);
	}

	return count;
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
	, Triangle* triangles
	//, OctreeNode* dev_octree
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
		glm::vec2 tmp_uv;

		/*Geom potentialIntersections[MAX_POTENTIAL_INTERSECTIONS];
		int potentialCount = 0;
		intersectOctree(dev_octree, pathSegment.ray, triangles, potentialIntersections, potentialCount);
		
		for (int i = 0; i < potentialCount; i++)
		{
			Geom& geom = potentialIntersections[i];*/
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
			else if (geom.type == OBJMESH) {
				if (checkMeshhBoundingBox(geom, pathSegment.ray))
				{
					t = triangleIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, triangles, geom.triangleIdStart, geom.triangleIdEnd, tmp_normal, outside, tmp_uv);
				}
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
			pathSegments[path_index].remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			//intersections[path_index].materialId = potentialIntersections[hit_geom_index].materialid;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			/*intersections[path_index].uv = uv;
			intersections[path_index].textureId = geoms[hit_geom_index].textureId;*/
		}
	}
}

__global__ void shadeBSDFMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* lights
	, int lightNum
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	PathSegment& path = pathSegments[idx];

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
				path.color *= (materialColor * material.emittance);
				path.remainingBounces = 0.0f;
			}
			else {
				glm::vec3 intersectionPoint = glm::normalize(path.ray.direction) * intersection.t + path.ray.origin;
				scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng, lights, lightNum);

#if DIRECT_LIGHT
				//Simple version of direct lighting
				if (path.remainingBounces == 2)
				{
					thrust::uniform_real_distribution<float> u01(0, 1);
					thrust::uniform_real_distribution<int> u02(0, lightNum - 1);
					Geom light = lights[u02(rng)];
					glm::vec3 sampledLight = glm::vec3(light.transform * glm::vec4(u01(rng), u01(rng), u01(rng), 1.f));
					path.ray.direction = glm::normalize(sampledLight - path.ray.origin);
					return;
				}
#endif
				//path.remainingBounces--;
			}
		}
		else {
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0.0f;
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

struct isPathCompleted
{
	__host__ __device__ bool operator()(const PathSegment& pathSegment) {
		return pathSegment.remainingBounces > 0;
	}
};

struct dev_materialIds
{
	__host__ __device__ bool operator()(const ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2)
	{
		return intersect1.materialId < intersect2.materialId;
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
	int oriPaths = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete && depth < traceDepth) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if	CACHE_FIRST_BOUNCE && ! ANTI_ALIASING
		if (depth == 0 && iter == 1) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles
				//, dev_octree
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_cache_intersections, dev_intersections,
				pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);

		}
		else if (depth == 0) {
			cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);

		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles
				//, dev_octree
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_triangles
			//, dev_octree
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
#if SORT_MATERIAL
		thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, dev_materialIds());
#endif
		shadeBSDFMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			lights,
			hst_scene->lightNum
			);
		cudaDeviceSynchronize();
#if STREAM_COMPACTION
		PathSegment* path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isPathCompleted());
		num_paths = path_end - dev_paths;
		iterationComplete = (num_paths == 0);
#endif
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (oriPaths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
