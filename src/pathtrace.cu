#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include<thrust/scan.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))


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

__device__ inline bool isNAN(const glm::vec3& v)
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

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, bool ShowGbuffer, bool ShowNormal, bool ShowPosition) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		if (ShowGbuffer && !ShowNormal && !ShowPosition) {
			float timeToIntersect = gBuffer[index].t * 256.0;

			pbo[index].w = 0;
			pbo[index].x = timeToIntersect;
			pbo[index].y = timeToIntersect;
			pbo[index].z = timeToIntersect;
		}
		else if (ShowNormal && !ShowPosition && !ShowGbuffer) {
			glm::vec3 Normal = abs(gBuffer[index].nor);
			Normal *= 255.0;

			pbo[index].w = 0;
			pbo[index].x = Normal.x;
			pbo[index].y = Normal.y;
			pbo[index].z = Normal.z;
		}
		else if (ShowPosition && !ShowNormal && !ShowGbuffer) {
			glm::vec3 Position = abs(gBuffer[index].pos) / 12.f;
			Position *= 255.0;

			pbo[index].w = 0;
			pbo[index].x = Position.x;
			pbo[index].y = Position.y;
			pbo[index].z = Position.z;
		}
	}
}

class CUDATimer
{
public:
	CUDATimer(const std::string& inName)
	{
		name = inName;

		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);
	}

	~CUDATimer()
	{
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);
	}

	void start()
	{
		cudaEventRecord(startEvent);
	}

	void stop()
	{
		cudaEventRecord(stopEvent);

		cudaEventSynchronize(stopEvent);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

		printf("%sIt takes: %f ms\n", name.c_str(), milliseconds);
	}

private:
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	std::string name;
};

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static int* dev_materialSortBuffer = nullptr;
static int* dev_materialSortBuffer2 = nullptr;
static ShadeableIntersection* dev_intersections = NULL;

int samplesPerPixel = 1;

bool sortMaterial = true;
bool Compaction = false;
bool CacheFirstBound = true;

int* dev_perm_x = nullptr;
int* dev_perm_y = nullptr;
int* dev_perm_z = nullptr;

perlin** dev_perlinNoise = nullptr;

bool perlinInitialized = false;

// static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_cache = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static glm::vec3* dev_image_denoised = NULL;
static glm::vec3* dev_image_denoise_temp = NULL;
void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

__global__ void initPerlin(perlin** perlinNoise, thrust::default_random_engine rng, int* dev_perm_x, int* dev_perm_y, int* dev_perm_z)
{
	*perlinNoise = new perlin(rng, dev_perm_x, dev_perm_y, dev_perm_z);
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	samplesPerPixel = guiData->SamplePerPixel;

	//hst_scene->state.traceDepth = guiData->Depth;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, samplesPerPixel * pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_materialSortBuffer, samplesPerPixel * pixelcount * sizeof(int));
	cudaMalloc(&dev_materialSortBuffer2, samplesPerPixel * pixelcount * sizeof(int));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection));

	// initialize any extra device memeory you need
	cudaMalloc(&dev_cache, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache, 0, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_perm_x, 256 * sizeof(int));
	cudaMemset(dev_perm_x, 0, 256 * sizeof(int));

	cudaMalloc(&dev_perm_y, 256 * sizeof(int));
	cudaMemset(dev_perm_y, 0, 256 * sizeof(int));

	cudaMalloc(&dev_perm_z, 256 * sizeof(int));
	cudaMemset(dev_perm_z, 0, 256 * sizeof(int));

	cudaMalloc(&dev_perlinNoise, sizeof(perlin*));

	thrust::default_random_engine rng = makeSeededRandomEngine(0, 1, 2);

	initPerlin<<<1, 1>>>(dev_perlinNoise, rng, dev_perm_x, dev_perm_y, dev_perm_z);

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
	cudaMalloc(&dev_image_denoised, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_denoised, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_image_denoise_temp, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_denoise_temp, 0, pixelcount * sizeof(glm::vec3));

	checkCUDAError("pathtraceInit");
}

__device__ glm::vec2 ConcentricSampleDisk(thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float r1 = 2.0f * u01(rng) - 1.0f;
	float r2 = 2.0f * u01(rng) - 1.0f;

	if (r1 == 0.0f && r2 == 0.0f) {
		return glm::vec2(0, 0);
	}

	float r, theta;
	if (fabs(r1) > fabs(r2)) {
		r = r1;
		theta = (3.1415926535 / 4.0f) * (r2 / r1);
	}
	else {
		r = r2;
		theta = (3.1415926535 / 2.0f) - (r1 / r2) * (3.1415926535 / 4.0f);
	}

	return r * glm::vec2(cos(theta), sin(theta));
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_materialSortBuffer);
	cudaFree(dev_materialSortBuffer2);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cache);
	cudaFree(dev_image_denoised);
	cudaFree(dev_image_denoise_temp);
	cudaFree(dev_gBuffer);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int samplesPerPixel)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);


		// implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		for (int i = 0; i < samplesPerPixel; i++)
		{
			int jitterIndex = cam.resolution.x * cam.resolution.y * i + index;

			PathSegment& segment = pathSegments[jitterIndex];

			segment.ray.origin = cam.position;
			segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

			glm::vec2 jitter = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));

			if (i == 0)
			{
				jitter = glm::vec2(0.0f);
			}

			pathSegments[jitterIndex].ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter[0])
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter[1]));

			if (cam.aperture > 0) {
				glm::vec2 pLens = cam.aperture * ConcentricSampleDisk(rng);
				float ft = cam.focalDistance / glm::abs(segment.ray.direction.z); //abs handel left and right hand coordinate system where z value might be negative
				glm::vec3 pFocus = segment.ray.direction * ft + segment.ray.origin;

				glm::vec3 newOri = pLens.x * cam.right + pLens.y * cam.up + cam.position;
				segment.ray.origin = newOri;
				segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
			}


			segment.pixelIndex = index;
			segment.remainingBounces = traceDepth;
		}
	}
}





__global__ void computeIntersections(
	int depth
	, int maxDepth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, glm::vec3* image
	, cudaTextureObject_t skyboxTex
	, int imageWidth
	, int imageHeight
	, int samplesPerPixel
	, int* materialKeys
)
{
	int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIndex >= num_paths)
	{
		return;
	}

	for (int i = 0; i < samplesPerPixel; i++)
	{
		int jitterIndex = imageWidth * imageHeight * i + pathIndex;

		PathSegment& pathSegment = pathSegments[jitterIndex];

		intersections[jitterIndex].materialId = -1;

		float t;
		glm::vec3 intersectPoint;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		float u;
		float v;

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
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, u, v);
			}
			else if (geom.type == MESH)
			{
				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == PROCEDURAL)
			{
				t = proceduralIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersectPoint = tmp_intersect;
				normal = tmp_normal;
			}
		}

		int materialId = -1;
		if (hit_geom_index == -1)
		{
			intersections[jitterIndex].t = -1.0f;

			if (pathSegment.remainingBounces == maxDepth)
			{
				pathSegment.needSkyboxColor = true;
			}

			pathSegment.remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			intersections[jitterIndex].t = t_min;
			intersections[jitterIndex].materialId = geoms[hit_geom_index].materialid;
			intersections[jitterIndex].surfaceNormal = normal;
			intersections[jitterIndex].frontFace = outside;
			intersections[jitterIndex].point = intersectPoint;
			intersections[jitterIndex].u = u;
			intersections[jitterIndex].v = v;
			materialId = intersections[jitterIndex].materialId;
			
		}
		materialKeys[jitterIndex] = materialId;
	}
}

__global__ void shadeFakeMaterial(
	int iter
	, int numPaths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int imageWidth
	, int imageHeight
	, int samplesPerPixel
	, perlin** perlinNoise
	, int* dev_perm_x
	, int* dev_perm_y
	, int* dev_perm_z
	, cudaTextureObject_t skyBoxTexture
)
{
	int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIndex >= numPaths)
	{
		return;
	}

	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathIndex, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	// getPointOnRay(pathSegments[idx].ray, intersection.t)

	glm::vec3 color = glm::vec3(0.0f);

	for (int i = 0; i < samplesPerPixel; i++)
	{
		int jitterIndex = imageWidth * imageHeight * i + pathIndex;

		if (pathSegments[jitterIndex].remainingBounces == 0)
		{

			glm::vec2 uv = sampleHDRMap(glm::normalize(pathSegments[jitterIndex].ray.direction));
			float4 skyColorRGBA = tex2D<float4>(skyBoxTexture, uv.x, uv.y);
			glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);

			color += pathSegments[jitterIndex].color;

			continue;
		}

		ShadeableIntersection intersection = shadeableIntersections[jitterIndex];

		Material material;

		if (intersection.materialId >= 0)
		{
			material = materials[intersection.materialId];
		}

		scatterRay(pathSegments[jitterIndex], intersection, intersection.point, intersection.surfaceNormal, intersection.frontFace, material, rng);

		if (material.pattern == Pattern::PerlinNoise)
		{
			glm::vec3 perlinNoiseColor = glm::vec3(1.0f, 1.0f, 1.0f) * 0.5f *
				(1.0f + glm::sin(1.0f * intersection.point.z + 10.0f *
					(*perlinNoise)->turb(intersection.point, 7, dev_perm_x, dev_perm_y, dev_perm_z)));


			glm::vec3 deepBlue = glm::vec3(0.0f, 0.0f, 0.5f);
			glm::vec3 lightBlue = glm::vec3(0.53f, 0.81f, 0.98f);
			glm::vec3 turquoise = glm::vec3(0.25f, 0.88f, 0.82f);
			glm::vec3 seaGreen = glm::vec3(0.13f, 0.55f, 0.13f);

			glm::vec3 intermediateColor1 = glm::mix(deepBlue, lightBlue, perlinNoiseColor.x);
			glm::vec3 intermediateColor2 = glm::mix(turquoise, seaGreen, perlinNoiseColor.y);
			glm::vec3 waterColor = glm::mix(intermediateColor1, intermediateColor2, 0.5f);

			pathSegments[jitterIndex].color *= waterColor;
		}


		color += pathSegments[jitterIndex].color;
	}

	color /= samplesPerPixel;

	pathSegments[pathIndex].color = color;
}


__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].t = shadeableIntersections[idx].t;
		gBuffer[idx].pos = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * shadeableIntersections[idx].t;
		gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
	}
}

__global__ void DenoiseKernel(glm::ivec2 resolution, glm::vec3* input, glm::vec3* output, GBufferPixel* gBuffer,
	float c_phi, float n_phi, float p_phi, float stepwidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	float kernel[25] = {
		0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f,
		0.0156f, 0.0625f, 0.0938f, 0.0625f, 0.0156f,
		0.0234f, 0.0938f, 0.1406f, 0.0938f, 0.0234f,
		0.0156f, 0.0625f, 0.0938f, 0.0625f, 0.0156f,
		0.0039f, 0.0156f, 0.0234f, 0.0156f, 0.0039f };

	glm::ivec2 offset[25] = {
		{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
		{-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
		{-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},
		{-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},
		{-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}
	};

	int idx = x + (y * resolution.x);

	glm::vec3 sum = glm::vec3(0.f);

	glm::vec3 cval = input[idx];
	glm::vec3 nval = gBuffer[idx].nor;
	glm::vec3 pval = gBuffer[idx].pos;

	float cum_w = 0.0f;

	for (int i = 0; i < 25; i++)
	{
		glm::ivec2 uv = glm::ivec2(x + offset[i].x * stepwidth, y + offset[i].y * stepwidth);

		int idxtmp = uv.x + uv.y * resolution.x;
		glm::vec3 ctmp = input[idxtmp];
		glm::vec3 t = cval - ctmp;
		float dist2 = glm::dot(t, t);
		float c_w = glm::min(glm::exp(-dist2 / c_phi), 1.0f);

		glm::vec3 ntmp = gBuffer[idxtmp].nor;
		t = nval - ntmp;
		dist2 = glm::max(glm::dot(t, t) / (stepwidth * stepwidth), 0.0f);
		float n_w = glm::min(glm::exp(-dist2 / n_phi), 1.0f);

		glm::vec3 ptmp = gBuffer[idxtmp].pos;
		t = pval - ptmp;
		dist2 = glm::dot(t, t);
		float p_w = glm::min(glm::exp(-dist2 / p_phi), 1.0f);

		float weight = c_w * n_w * p_w;

		sum += ctmp * weight * kernel[i];
		cum_w += weight * kernel[i];
	}

	output[idx] = sum / cum_w;
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int numPaths, glm::vec3* image, PathSegment* iterationPaths, cudaTextureObject_t skyboxTexture)
{
	int pathIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (pathIndex < numPaths)
	{
		PathSegment iterationPath = iterationPaths[pathIndex];
		if (!isNAN(iterationPath.color))
		{
			glm::vec2 uv = sampleHDRMap(glm::normalize(iterationPath.ray.direction));
			float4 skyColorRGBA = tex2D<float4>(skyboxTexture, uv.x, uv.y);
			glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);

			if (iterationPath.needSkyboxColor)
			{
				image[iterationPath.pixelIndex] += iterationPath.color * skyColor * 3.0f;
			}
			else
			{
				image[iterationPath.pixelIndex] += iterationPath.color;
			}
		}
	}
}

__global__ void computeReflectionForWaterPattern(
	int num_paths,
	PathSegment* dev_paths,
	ShadeableIntersection* dev_intersections,
	Material* dev_materials
) {
	int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIndex >= num_paths) return;

	ShadeableIntersection intersection = dev_intersections[pathIndex];
	if (intersection.t > 0.0f) {  // if there is an intersection
		Material material = dev_materials[intersection.materialId];
		if (material.pattern == Pattern::PerlinNoise) {
			glm::vec3 incidentDir = -dev_paths[pathIndex].ray.direction;
			glm::vec3 reflectDir = glm::reflect(incidentDir, intersection.surfaceNormal);

			// Update the path segment with the new direction
			dev_paths[pathIndex].ray.direction = reflectDir;
		}
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
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, samplesPerPixel);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int current_num_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	bool iterationComplete = false;

	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		if (depth == 0 && CacheFirstBound)
		{
			if (iter == 1) {
				// tracing
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, traceDepth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, dev_image
					, hst_scene->skyboxTextureObject
					, cam.resolution.x
					, cam.resolution.y
					, samplesPerPixel
					, dev_materialSortBuffer
					);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();
				cudaMemcpy(dev_cache, dev_intersections, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				cudaMemcpy(dev_intersections, dev_cache, samplesPerPixel * pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
			printf("generateGBuffer\n");
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, traceDepth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_image
				, hst_scene->skyboxTextureObject
				, cam.resolution.x
				, cam.resolution.y
				, samplesPerPixel
				, dev_materialSortBuffer
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		depth++;

		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			cam.resolution.x,
			cam.resolution.y,
			samplesPerPixel,
			dev_perlinNoise,
			dev_perm_x,
			dev_perm_y,
			dev_perm_z,
			hst_scene->skyboxTextureObject
			);

		//CUDATimer timer("shadeFakeMaterial");

		//timer.start();

		if (sortMaterial)
		{
			cudaMemcpy(dev_materialSortBuffer2, dev_materialSortBuffer, sizeof(int) * num_paths, cudaMemcpyDeviceToDevice);
			thrust::sort_by_key(thrust::device, dev_materialSortBuffer, dev_materialSortBuffer + num_paths, dev_intersections);
			thrust::sort_by_key(thrust::device, dev_materialSortBuffer2, dev_materialSortBuffer2 + num_paths, dev_paths);

			//timer.stop();
			if (Compaction)
			{
				dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, dev_path_end, returnRemainBounce());
				current_num_paths = dev_path_end - dev_paths;
			}

			//printf("%d\n", current_num_paths);

			iterationComplete = (depth >= traceDepth || current_num_paths <= 0);
		}
		else
		{
			if (depth == traceDepth)
			{
				iterationComplete = true; 
			}
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;

			samplesPerPixel = guiData->SamplePerPixel;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths, hst_scene->skyboxTextureObject);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

void showGBuffer(uchar4* pbo, bool ShowGbuffer, bool ShowNormal, bool ShowPosition) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer, ShowGbuffer, ShowNormal, ShowPosition);
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void denoise(uchar4* pbo, int iter,
	float colorWeight, float normalWeight,
	float positionWeight, float filterSize)
{
	// Initialization and copying from denoise function
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	cudaMemcpy(dev_image_denoise_temp, dev_image, cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	for (int i = 1; i < filterSize; i *= 2)
	{
		DenoiseKernel << <blocksPerGrid2d, blockSize2d >> > (cam.resolution,
			dev_image_denoise_temp,
			dev_image_denoised,
			dev_gBuffer,
			colorWeight, normalWeight, positionWeight, i);

		cudaDeviceSynchronize();

		glm::vec3* temp = dev_image_denoised;
		dev_image_denoise_temp = dev_image_denoised;
		dev_image_denoised = temp;
	}

	cudaMemcpy(hst_scene->state.image.data(), dev_image_denoised, cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution,
		iter, dev_image_denoised);
}
