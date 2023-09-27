#include "device_launch_parameters.h"

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
#include <glm/gtx/projection.hpp>
#include "utilities.h"
#include "extrautils.hpp"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "toggles.h"

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

static Triangle* dev_tris = NULL;

static BvhNode* dev_bvh_nodes = NULL;
static int* dev_bvh_tri_idx = NULL;

static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static thrust::device_ptr<PathSegment> dev_thrust_paths = NULL;
static thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections = NULL;

static int numTextures;
static cudaTextureObject_t* host_textureObjects = NULL; // array owned by host
static cudaTextureObject_t* dev_textureObjects = NULL; // array owned by device
static cudaArray_t* host_textureArrayPtrs = NULL; // array owned by host

#if FIRST_BOUNCE_CACHE
static bool fbcNeedsRefresh = true;
static ShadeableIntersection* dev_intersections_fbc = NULL;
static thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections_fbc = NULL;
#endif


void Pathtracer::InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void Pathtracer::init(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_nodes, scene->bvhNodes.size() * sizeof(BvhNode));
	cudaMemcpy(dev_bvh_nodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_bvh_tri_idx, scene->bvhTriIdx.size() * sizeof(int));
	cudaMemcpy(dev_bvh_tri_idx, scene->bvhTriIdx.data(), scene->bvhTriIdx.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);

	initTextures(scene);

#if FIRST_BOUNCE_CACHE
	cudaMalloc(&dev_intersections_fbc, pixelcount * sizeof(ShadeableIntersection));
	dev_thrust_intersections_fbc = thrust::device_pointer_cast(dev_intersections_fbc);
#endif

	checkCUDAError("pathtraceInit");
}

void Pathtracer::initTextures(Scene* scene)
{
	numTextures = scene->textures.size();
	if (numTextures == 0)
	{
		return;
	}

	host_textureObjects = new cudaTextureObject_t[numTextures];
	host_textureArrayPtrs = new cudaArray_t[numTextures];

	for (int i = 0; i < numTextures; ++i)
	{
		const Texture& texture = scene->textures[i];

		// wasn't working with linear memory so changed to array
		// https://stackoverflow.com/questions/63408787/texture-object-fetching-in-cuda
		auto channelDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(&host_textureArrayPtrs[i], &channelDesc, texture.width, texture.height);
		cudaMemcpy2DToArray(host_textureArrayPtrs[i], 0, 0, texture.host_dataPtr, texture.width * sizeof(uchar4), texture.width * sizeof(uchar4), texture.height, cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = host_textureArrayPtrs[i];

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeNormalizedFloat;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&host_textureObjects[i], &resDesc, &texDesc, NULL);
	}

	cudaMalloc((void**)&dev_textureObjects, numTextures * sizeof(cudaTextureObject_t));
	cudaMemcpy(dev_textureObjects, host_textureObjects, numTextures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
}

void Pathtracer::free() {
	cudaFree(dev_image);  // no-op if dev_image is null

	cudaFree(dev_geoms);
	cudaFree(dev_materials);

	cudaFree(dev_tris);

	cudaFree(dev_bvh_nodes);
	cudaFree(dev_bvh_tri_idx);

	cudaFree(dev_paths);
	cudaFree(dev_intersections);

	freeTextures();

#if FIRST_BOUNCE_CACHE
	cudaFree(dev_intersections_fbc);
#endif

	checkCUDAError("pathtraceFree");
}

void Pathtracer::freeTextures()
{
	if (numTextures == 0)
	{
		return;
	}

	cudaFree(dev_textureObjects);

	for (int i = 0; i < numTextures; ++i)
	{
		cudaDestroyTextureObject(host_textureObjects[i]);
		cudaFreeArray(host_textureArrayPtrs[i]);
	}

	delete[] host_textureObjects;
	delete[] host_textureArrayPtrs;
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

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}

	int index = x + (y * cam.resolution.x);
	PathSegment& segment = pathSegments[index];

	segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 noLensDirection = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
	);

	if (cam.lensRadius > 0)
	{
		float z = glm::length(glm::proj(noLensDirection, cam.view));
		glm::vec3 pFocus = cam.position + (noLensDirection * cam.focusDistance / z);
		glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));

		Ray newRay;
		newRay.origin = cam.position + (pLens.x * cam.right + pLens.y * cam.up);
		newRay.direction = glm::normalize(pFocus - newRay.origin);
		segment.ray = newRay;
	}
	else
	{
		segment.ray = { cam.position, noLensDirection };
	}

	segment.pixelIndex = index;
	segment.bouncesSoFar = 0;
	segment.remainingBounces = traceDepth;
}

__global__ void computeIntersections(
	int depth, 
	int num_paths, 
	PathSegment* pathSegments, 
	Geom* geoms,
	int geoms_size,
	Triangle* tris,
	BvhNode* bvhNodes,
	int* bvhTriIdx,
	ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths)
	{
		return;
	}

	PathSegment pathSegment = pathSegments[path_index];

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	glm::vec2 uv;
	int triIdx;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	glm::vec2 tmp_uv;
	int tmp_triIdx;

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++)
	{
		Geom& geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
		}
		else if (geom.type == MESH)
		{
			t = meshIntersectionTest(geom, tris, bvhNodes, bvhTriIdx, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_triIdx);
		}

		if (t < 0 || t > t_min)
		{
			continue;
		}

		t_min = t;
		hit_geom_index = i;
		intersect_point = tmp_intersect;
		normal = tmp_normal;
		uv = tmp_uv;
		triIdx = tmp_triIdx;
	}

	if (hit_geom_index == -1)
	{
		intersections[path_index].t = -1.0f;
	}
	else
	{
		//The ray hits something
		ShadeableIntersection& isect = intersections[path_index];
		isect.hitGeomIdx = hit_geom_index;
		isect.t = t_min;
		isect.materialId = geoms[hit_geom_index].materialId;
		isect.surfaceNormal = normal;
		isect.uv = uv;
		isect.triIdx = triIdx;
	}
}

struct SegmentProcessingSettings
{
	bool russianRoulette;
};

__device__ void processSegment(
	PathSegment& segment, 
	ShadeableIntersection& intersection,
	Geom* geoms,
	Triangle* tris,
	Material* materials, 
	cudaTextureObject_t* textureObjects, 
	int iter, 
	int idx, 
	SegmentProcessingSettings settings)
{
	if (intersection.t <= 0.0f)
	{
		segment.color = glm::vec3(0.0f);
		segment.remainingBounces = 0;
		return;
	}

	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.bouncesSoFar + 1);

	Material material = materials[intersection.materialId];

	if (material.emission.strength > 0)
	{
		segment.color *= material.emission.color * material.emission.strength;
		segment.remainingBounces = 0;
		return;
	} 
	
	scatterRay(
		segment,
		intersection,
		getPointOnRay(segment.ray, intersection.t),
		geoms,
		tris,
		material,
		textureObjects,
		rng
	);

#if DEBUG_SHOW_NORMALS
		segment.color = (intersection.surfaceNormal + 1.f) / 2.f;
		segment.remainingBounces = 0;
		return;
#endif

	++segment.bouncesSoFar;

	if (--segment.remainingBounces == 0)
	{
		segment.color = glm::vec3(0.0f);
		return;
	}

	if (settings.russianRoulette && segment.bouncesSoFar > 3)
	{
		float q = glm::max(0.05f, 1 - Utils::luminance(segment.color));
		if (u01(rng) < q)
		{
			segment.color = glm::vec3(0.0f);
			segment.remainingBounces = 0;
			return;
		}

		segment.color /= (1 - q);
	}
}

__global__ void shadeMaterial(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Geom* geoms,
	Triangle* tris,
	Material* materials,
	cudaTextureObject_t* textureObjects,
	SegmentProcessingSettings settings
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
	{
		return;
	}

	PathSegment segment = pathSegments[idx];
	processSegment(segment, shadeableIntersections[idx], geoms, tris, materials, textureObjects, iter, idx, settings);

	pathSegments[idx] = segment;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];

#if DEBUG_NAN_MAGENTA
		if (isnan(iterationPath.color.x) || isnan(iterationPath.color.y) || isnan(iterationPath.color.z))
		{
			image[iterationPath.pixelIndex] = glm::vec3(1, 0, 1);
		}
		else
		{
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
#else
	image[iterationPath.pixelIndex] += iterationPath.color;
#endif
	}
}

struct partition_predicate
{
	__host__ __device__ bool operator()(const PathSegment& ps)
	{
		return ps.remainingBounces > 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void Pathtracer::pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 512;

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_total_paths = dev_path_end - dev_paths;
	int num_valid_paths;

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	num_valid_paths = num_total_paths;

	while (num_valid_paths > 0) {
		// tracing
		dim3 numblocksPathSegmentTracing = (num_valid_paths + blockSize1d - 1) / blockSize1d;

#if FIRST_BOUNCE_CACHE
		if (guiData->firstBounceCache && !fbcNeedsRefresh && depth == 0)
		{
			thrust::copy(
				thrust::device,
				dev_thrust_intersections_fbc,
				dev_thrust_intersections_fbc + num_total_paths,
				dev_thrust_intersections
			);
		} 
		else
#endif
		{
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth,
				num_valid_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_tris,
				dev_bvh_nodes,
				dev_bvh_tri_idx,
				dev_intersections
			);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		++depth;

		if (guiData->sortByMaterial)
		{
			thrust::sort_by_key(
				thrust::device,
				dev_thrust_intersections,
				dev_thrust_intersections + num_valid_paths,
				dev_thrust_paths
			);
			checkCUDAError("sort by material");
		}

		SegmentProcessingSettings settings;
		settings.russianRoulette = guiData->russianRoulette;

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_valid_paths,
			dev_intersections,
			dev_paths,
			dev_geoms,
			dev_tris,
			dev_materials,
			dev_textureObjects,
			settings
		);
		checkCUDAError("shade material");

#if FIRST_BOUNCE_CACHE
		if (guiData->firstBounceCache && depth == 1 && fbcNeedsRefresh)
		{
			thrust::copy(
				thrust::device,
				dev_thrust_intersections,
				dev_thrust_intersections + num_total_paths,
				dev_thrust_intersections_fbc
			);
			checkCUDAError("first bounce cache");
			fbcNeedsRefresh = false;
		}
#endif

		thrust::device_ptr<PathSegment> middle = thrust::partition(
			thrust::device,
			dev_thrust_paths,
			dev_thrust_paths + num_valid_paths,
			partition_predicate()
		);
		checkCUDAError("partition");

		num_valid_paths = middle - dev_thrust_paths;

		if (guiData != NULL)
		{
			guiData->tracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_total_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

void Pathtracer::onCamChanged()
{
#if FIRST_BOUNCE_CACHE
	fbcNeedsRefresh = true;
#endif
}