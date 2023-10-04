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

#include <OpenImageDenoise/oidn.h>

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
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__host__ __device__ glm::vec3 ACESFilm(glm::vec3 x)
{
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}

__host__ __device__ glm::vec3 hdrToLdr(glm::vec3 col)
{
	return glm::pow(ACESFilm(col), glm::vec3(INVERSE_GAMMA));
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image, bool isNormals) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index] / (float)iter;

		if (isNormals)
		{
			pix = (pix + 1.f) / 2.f;
		}
		else
		{
			pix = hdrToLdr(pix);
		}

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image_raw = NULL;
static glm::vec3* dev_image_final = NULL;

static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;

static Triangle* dev_tris = NULL;

static BvhNode* dev_bvh_nodes = NULL;

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

static OIDNDevice oidnDevice;
static OIDNFilter oidnFilterAlbedo;
static OIDNFilter oidnFilter;

static OIDNBuffer oidnColorBuf;
static OIDNBuffer oidnAlbedoBuf;
static OIDNBuffer oidnNormalBuf;
static OIDNBuffer oidnOutputBuf;

static glm::vec3* dev_first_hit_albedo_accum = NULL;
static glm::vec3* dev_first_hit_albedo = NULL;
static glm::vec3* dev_first_hit_normals_accum = NULL;
static glm::vec3* dev_first_hit_normals = NULL;

void checkOIDNError()
{
#if ERRORCHECK
	const char* errorMessage;
	if (oidnGetDeviceError(oidnDevice, &errorMessage) != OIDN_ERROR_NONE)
	{
		printf("Error: %s\n", errorMessage);
		exit(EXIT_FAILURE);
	}
#endif
}

void Pathtracer::InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void Pathtracer::init(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelCount = cam.resolution.x * cam.resolution.y;

	auto imageSizeBytes = pixelCount * sizeof(glm::vec3);
	cudaMalloc(&dev_image_raw, imageSizeBytes);
	cudaMemset(dev_image_raw, 0, imageSizeBytes);
	cudaMalloc(&dev_image_final, imageSizeBytes);
	cudaMemset(dev_image_final, 0, imageSizeBytes);

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_nodes, scene->bvhNodes.size() * sizeof(BvhNode));
	cudaMemcpy(dev_bvh_nodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_paths, pixelCount * sizeof(PathSegment));
	dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
	cudaMalloc(&dev_intersections, pixelCount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelCount * sizeof(ShadeableIntersection));
	dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);

#if FIRST_BOUNCE_CACHE
	cudaMalloc(&dev_intersections_fbc, pixelcount * sizeof(ShadeableIntersection));
	dev_thrust_intersections_fbc = thrust::device_pointer_cast(dev_intersections_fbc);
#endif

	cudaMalloc(&dev_first_hit_albedo_accum, imageSizeBytes);
	cudaMemset(dev_first_hit_albedo_accum, 0, imageSizeBytes);
	cudaMalloc(&dev_first_hit_albedo, imageSizeBytes);
	cudaMemset(dev_first_hit_albedo, 0, imageSizeBytes);
	cudaMalloc(&dev_first_hit_normals_accum, imageSizeBytes);
	cudaMemset(dev_first_hit_normals_accum, 0, imageSizeBytes);
	cudaMalloc(&dev_first_hit_normals, imageSizeBytes);
	cudaMemset(dev_first_hit_normals, 0, imageSizeBytes);

	initTextures();

	initOIDN();

	checkCUDAError("pathtraceInit");
}

void Pathtracer::initTextures()
{
	numTextures = hst_scene->textures.size();
	if (numTextures == 0)
	{
		return;
	}

	host_textureObjects = new cudaTextureObject_t[numTextures];
	host_textureArrayPtrs = new cudaArray_t[numTextures];

	for (int i = 0; i < numTextures; ++i)
	{
		const Texture& texture = hst_scene->textures[i];

		// wasn't working with linear memory so changed to array
		// https://stackoverflow.com/questions/63408787/texture-object-fetching-in-cuda
		auto channelDesc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&host_textureArrayPtrs[i], &channelDesc, texture.width, texture.height);
		cudaMemcpy2DToArray(host_textureArrayPtrs[i], 0, 0, texture.host_dataPtr, texture.width * sizeof(float4), texture.width * sizeof(float4), texture.height, cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = host_textureArrayPtrs[i];

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&host_textureObjects[i], &resDesc, &texDesc, NULL);
	}

	cudaMalloc((void**)&dev_textureObjects, numTextures * sizeof(cudaTextureObject_t));
	cudaMemcpy(dev_textureObjects, host_textureObjects, numTextures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
}

void Pathtracer::initOIDN()
{
	int deviceId = -1;
	cudaStream_t stream = NULL;

	oidnDevice = oidnNewCUDADevice(&deviceId, &stream, 1);
	oidnCommitDevice(oidnDevice);

	const Camera& cam = hst_scene->state.camera;

	const int pixelCount = cam.resolution.x * cam.resolution.y;
	const int imageSizeBytes = pixelCount * sizeof(glm::vec3);
	oidnColorBuf = oidnNewSharedBuffer(oidnDevice, dev_image_raw, imageSizeBytes);
	oidnAlbedoBuf = oidnNewSharedBuffer(oidnDevice, dev_first_hit_albedo, imageSizeBytes);
	oidnNormalBuf = oidnNewSharedBuffer(oidnDevice, dev_first_hit_normals, imageSizeBytes);
	oidnOutputBuf = oidnNewSharedBuffer(oidnDevice, dev_image_final, imageSizeBytes);

	oidnFilter = oidnNewFilter(oidnDevice, "RT");
	oidnSetFilterImage(oidnFilter, "color", oidnColorBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterImage(oidnFilter, "albedo", oidnAlbedoBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterImage(oidnFilter, "normal", oidnNormalBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterImage(oidnFilter, "output", oidnOutputBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterBool(oidnFilter, "hdr", true);
	oidnSetFilterInt(oidnFilter, "quality", OIDN_QUALITY_BALANCED);
	oidnCommitFilter(oidnFilter);

	oidnFilterAlbedo = oidnNewFilter(oidnDevice, "RT");
	oidnSetFilterImage(oidnFilterAlbedo, "color", oidnAlbedoBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterImage(oidnFilterAlbedo, "output", oidnAlbedoBuf, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y, 0, 0, 0);
	oidnSetFilterBool(oidnFilterAlbedo, "hdr", true);
	oidnSetFilterInt(oidnFilterAlbedo, "quality", OIDN_QUALITY_BALANCED);
	oidnCommitFilter(oidnFilterAlbedo);

	checkOIDNError();
}

void Pathtracer::free() {
	cudaFree(dev_image_raw); // no-op if dev_image_raw is null

	cudaFree(dev_geoms);
	cudaFree(dev_materials);

	cudaFree(dev_tris);

	cudaFree(dev_bvh_nodes);

	cudaFree(dev_paths);
	cudaFree(dev_intersections);

#if FIRST_BOUNCE_CACHE
	cudaFree(dev_intersections_fbc);
#endif

	freeTextures();

	oidnReleaseBuffer(oidnColorBuf);
	oidnReleaseBuffer(oidnAlbedoBuf);
	oidnReleaseBuffer(oidnNormalBuf);
	oidnReleaseBuffer(oidnOutputBuf);

	cudaFree(dev_first_hit_albedo_accum);
	cudaFree(dev_first_hit_albedo);
	cudaFree(dev_first_hit_normals_accum);
	cudaFree(dev_first_hit_normals);

	oidnReleaseFilter(oidnFilterAlbedo);
	oidnReleaseFilter(oidnFilter);
	oidnReleaseDevice(oidnDevice);

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
__global__ void generateRayFromCamera(
	const Camera cam, 
	const int iter, 
	const int traceDepth, 
	PathSegment* pathSegments)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}

	const int index = x + (y * cam.resolution.x);
	PathSegment& segment = pathSegments[index];

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	const glm::vec3 noLensDirection = glm::normalize(cam.view
		- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
		- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
	);

	segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

	if (cam.lensRadius > 0)
	{
		const float z = glm::length(glm::proj(noLensDirection, cam.view));
		const glm::vec3 pFocus = cam.position + (noLensDirection * cam.focusDistance / z);
		const glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));

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
	segment.needsFirstHitData = true;
}

struct SegmentProcessingSettings
{
	bool russianRoulette;
	bool useBvh;
};

__global__ void computeIntersections(
	int depth, 
	int num_paths, 
	const PathSegment* const pathSegments, 
	const Geom* const geoms,
	int geoms_size,
	const Triangle* const tris,
	const BvhNode* const bvhNodes,
	ShadeableIntersection* intersections,
	const SegmentProcessingSettings settings)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths)
	{
		return;
	}

	const PathSegment& pathSegment = pathSegments[path_index];

	float t;
	float tMin = FLT_MAX;

	ShadeableIntersection newIsect;
	newIsect.hitGeomIdx = -1;

	glm::vec3 tmp_normal;
	glm::vec2 tmp_uv;
	int tmp_triIdx;

	for (int i = 0; i < geoms_size; ++i)
	{
		const Geom& geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_normal);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_normal);
		}
		else if (geom.type == MESH)
		{
			t = meshIntersectionTest(geom, tris, bvhNodes, pathSegment.ray, tmp_normal, tmp_uv, tmp_triIdx, tMin, settings.useBvh);
		}

		if (t < 0 || t > tMin)
		{
			continue;
		}

		tMin = t;
		newIsect.hitGeomIdx = i;
		newIsect.surfaceNormal = tmp_normal;
		newIsect.uv = tmp_uv;
		newIsect.triIdx = tmp_triIdx;
	}

	if (newIsect.hitGeomIdx == -1)
	{
		intersections[path_index].t = -1.0f;
		return;
	}

	newIsect.t = tMin;
	newIsect.materialId = geoms[newIsect.hitGeomIdx].materialId;
	intersections[path_index] = newIsect;
}

__device__ void processSegment(
	int iter,
	int idx,
	PathSegment& segment, 
	ShadeableIntersection& isect,
	const Geom* const geoms,
	const Triangle* const tris,
	const Material* const materials,
	const cudaTextureObject_t* const textureObjects,
	const SegmentProcessingSettings settings,
	int envMapTextureIdx,
	float envMapStrength)
{
	if (isect.t <= 0.0f)
	{
		glm::vec3 lightCol = glm::vec3(0.f);

		if (envMapTextureIdx != -1)
		{
			const glm::vec3 dir = segment.ray.direction;
			
			float theta = acosf(dir.y);
			float phi = atan2f(dir.z, dir.x);

			float u = (phi + PI) * ONE_OVER_TWO_PI;
			float v = theta * ONE_OVER_PI;

			lightCol = tex2DCustom3(textureObjects[envMapTextureIdx], glm::vec2(u, v)) * envMapStrength;

			segment.color *= lightCol;
		}
		else
		{
			segment.color = lightCol;
		}

		if (segment.needsFirstHitData)
		{
			segment.firstHitAlbedo = lightCol;
			segment.firstHitNormal = glm::vec3(0.0f);
			segment.needsFirstHitData = false;
		}

		segment.remainingBounces = 0;
		return;
	}

	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.bouncesSoFar + 1);

	Material m = materials[isect.materialId];
	
	bool wasEmission = scatterRay(
		segment,
		isect,
		getPointOnRay(segment.ray, isect.t),
		geoms,
		tris,
		m,
		textureObjects,
		rng
	);

	++segment.bouncesSoFar;
	
	if (--segment.remainingBounces <= 0)
	{
		if (!wasEmission)
		{
			segment.color = glm::vec3(0.0f);
		}

		if (segment.needsFirstHitData)
		{
			segment.firstHitAlbedo = segment.color;
			segment.firstHitNormal = isect.surfaceNormal;
			segment.needsFirstHitData = false;
		}

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
	const Geom* const geoms,
	const Triangle* const tris,
	const Material* const materials,
	const cudaTextureObject_t* const textureObjects,
	const SegmentProcessingSettings settings,
	int envMapTextureIdx,
	float envMapStrength
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
	{
		return;
	}

	PathSegment segment = pathSegments[idx];
	processSegment(iter, idx, segment, shadeableIntersections[idx], geoms, tris, materials, 
		textureObjects, settings, envMapTextureIdx, envMapStrength);
	pathSegments[idx] = segment;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, 
	glm::vec3* firstHitAlbedoAccum, glm::vec3* firstHitAlbedo,
	glm::vec3* firstHitNormalsAccum, glm::vec3* firstHitNormals, 
	PathSegment* iterationPaths, int iter)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nPaths)
	{
		return;
	}

	PathSegment iterationPath = iterationPaths[index];

	glm::vec3 col = glm::clamp(iterationPath.color, 0.f, 10.f);

#if DEBUG_NAN_MAGENTA
	if (isnan(col.x) || isnan(col.y) || isnan(col.z))
	{
		image[iterationPath.pixelIndex] = glm::vec3(1, 0, 1);
		printf("found a nan\n");
	}
	else
	{
		image[iterationPath.pixelIndex] += col;
	}
#else
	image[iterationPath.pixelIndex] += col;
#endif

	float mult = 1 / (float)iter;

	glm::vec3 albedoAccum = firstHitAlbedoAccum[iterationPath.pixelIndex] + iterationPath.firstHitAlbedo;
	firstHitAlbedoAccum[iterationPath.pixelIndex] = albedoAccum;
	firstHitAlbedo[iterationPath.pixelIndex] = albedoAccum * mult;

	glm::vec3 normalAccum = firstHitNormalsAccum[iterationPath.pixelIndex] + iterationPath.firstHitNormal;
	firstHitNormalsAccum[iterationPath.pixelIndex] = normalAccum;
	firstHitNormals[iterationPath.pixelIndex] = normalAccum * mult;
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

	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	const int blockSize1d = 768;

	int depth = 0;
	const PathSegment* dev_path_end = dev_paths + pixelcount;
	const int num_total_paths = dev_path_end - dev_paths;
	int num_valid_paths;

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	num_valid_paths = num_total_paths;

	while (num_valid_paths > 0) {
		dim3 numblocksPathSegmentTracing = (num_valid_paths + blockSize1d - 1) / blockSize1d;

		SegmentProcessingSettings settings;
		settings.russianRoulette = guiData->russianRoulette;
		settings.useBvh = guiData->useBvh;

#if FIRST_BOUNCE_CACHE
		if (guiData->firstBounceCache && !fbcNeedsRefresh && depth == 0)
		{
			thrust::copy(
				thrust::device,
				dev_thrust_intersections_fbc,
				dev_thrust_intersections_fbc + num_total_paths,
				dev_thrust_intersections
			);

			checkCUDAError("thrust copy from fbc intersections");
		} 
		else
#endif
		{
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
				depth,
				num_valid_paths,
				dev_paths,
				dev_geoms,
				hst_scene->geoms.size(),
				dev_tris,
				dev_bvh_nodes,
				dev_intersections,
				settings
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

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_valid_paths,
			dev_intersections,
			dev_paths,
			dev_geoms,
			dev_tris,
			dev_materials,
			dev_textureObjects,
			settings,
			cam.envMapTextureIdx,
			cam.envMapStrength
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
			checkCUDAError("thrust copy to fbc intersections");
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
	finalGather<<<numBlocksPixels, blockSize1d>>>(
		num_total_paths, 
		dev_image_raw,
		dev_first_hit_albedo_accum,
		dev_first_hit_albedo,
		dev_first_hit_normals_accum,
		dev_first_hit_normals,
		dev_paths,
		iter
	);
	checkCUDAError("final gather");

	///////////////////////////////////////////////////////////////////////////

	if (guiData->denoising)
	{
		if ((iter - 1) % guiData->denoiseInterval != 0)
		{
			return;
		}

		oidnExecuteFilter(oidnFilterAlbedo);
		oidnExecuteFilter(oidnFilter);
		checkOIDNError();
	}

	///////////////////////////////////////////////////////////////////////////

	glm::vec3* dev_image_ptr;
	if (guiData->showAlbedo)
	{
		dev_image_ptr = dev_first_hit_albedo_accum;
	}
	else if (guiData->showNormals)
	{
		dev_image_ptr = dev_first_hit_normals_accum;
	}
	else
	{
		dev_image_ptr = guiData->denoising ? dev_image_final : dev_image_raw;
	}

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
		pbo, 
		cam.resolution, 
		iter, 
		dev_image_ptr,
		guiData->showNormals
	);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image_ptr,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("denoise and send to OpenGL");
}

void Pathtracer::onCamChanged()
{
#if FIRST_BOUNCE_CACHE
	fbcNeedsRefresh = true;
#endif
}