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
#include "config.h"
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
#include "texture.h"
#include "mathUtil.h"

//#include "utilities.cuh"

#define USE_FIRST_BOUNCE_CACHE 0
#define USE_SORT_BY_MATERIAL 0
#define MIS 1
#define USE_BVH 1
#define TONE_MAPPING 0

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
		glm::vec3 pixel = image[index] / float(iter);
		glm::vec3 c;
		glm::ivec3 color;
#if TONE_MAPPING
		pixel = (pixel * (pixel * 2.51f + 0.03f)) / (pixel * (pixel * 2.43f + 0.59f) + 0.14f);
#endif
		// Gamma correction
		//pixel = glm::pow(pixel, glm::vec3(1.0f / 2.2f));
		
		// Map to 0-255
		color.x = glm::clamp((int)(pixel.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pixel.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pixel.z * 255.0), 0, 255);
		//color.x = glm::clamp((int)pix.x, 0, 255);
		//color.y = glm::clamp((int)pix.y, 0, 255);
		//color.z = glm::clamp((int)pix.z, 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static SceneConfig* scene_config = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Triangle* dev_triangles= nullptr;
static DevScene* scene;

/* TODO: Solve the weird shiny line in the empty room! */
//static Scene* hst_scene = new Scene("..\\scenes\\pathtracer_empty_room.glb");

//static Scene* hst_scene = new Scene("..\\scenes\\pathtracer_mis_demo.glb");
static Scene * hst_scene = new Scene("..\\scenes\\pathtracer_demo.glb");
static BSDFStruct * dev_bsdfStructs = nullptr;
static BVHAccel * bvh = nullptr;
static BVHNode* dev_bvhNodes = nullptr;
static ShadeableIntersection * dev_first_iteration_first_bounce_cache_intersections = nullptr;
static Texture * dev_textures = nullptr;
static TextureInfo * dev_textureInfos = nullptr;
static Light* dev_lights = nullptr;
int textureSize = 0;

#if USE_SORT_BY_MATERIAL
thrust::device_ptr<int> dev_bsdfStructIDs = nullptr;
thrust::device_ptr<int> dev_bsdfStructIDs_copy = nullptr;
#endif
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

__global__ void initDeviceTextures(Texture & dev_texture, const TextureInfo & dev_textureInfo, unsigned char * data) {
	dev_texture.width = dev_textureInfo.width;
	dev_texture.height = dev_textureInfo.height;
	dev_texture.nrChannels = dev_textureInfo.nrChannels;
	printf("dev_texture.width: %d, dev_texture.height: %d, dev_texture.nrChannels: %d\n", dev_texture.width, dev_texture.height, dev_texture.nrChannels);
	dev_texture.data = data;
	printf("dev_texture.data[0]: %d\n", dev_texture.data[0]);

	auto test_sample = sampleTextureRGB(dev_texture, glm::vec2(1.0f, 1.0f));
	printf("test_sample: %f, %f, %f, %f\n", test_sample.x, test_sample.y, test_sample.z);
}

__global__ void initPrimitivesNormalTexture(Triangle* triangles, Texture* textures, int triangle_size) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < triangle_size){
		triangles[index].normalTexture = textures + triangles[index].normalTextureID;
	}
}

__global__ void initBSDFWithTextures(BSDFStruct* bsdfStructs, Texture* texture, int bsdfStructs_size) {
	for (size_t i = 0; i < bsdfStructs_size; i++)
	{
		if (bsdfStructs[i].baseColorTextureID != -1)
			bsdfStructs[i].baseColorTexture	= &(texture[bsdfStructs[i].baseColorTextureID]);
		if (bsdfStructs[i].metallicRoughnessTextureID != -1)
			bsdfStructs[i].metallicRoughnessTexture = &(texture[bsdfStructs[i].metallicRoughnessTextureID]);
		if (bsdfStructs[i].normalTextureID != -1)
			bsdfStructs[i].normalTexture = &(texture[bsdfStructs[i].normalTextureID]);
		if (bsdfStructs[i].emissiveTextureID != -1)
			bsdfStructs[i].emissiveTexture = &(texture[bsdfStructs[i].emissiveTextureID]);
		printf("bsdfStructs[i].diffuseTextureID: %d\n", bsdfStructs[i].baseColorTextureID);
		printf("bsdfStructs[i].roughnessTextureID: %d\n", bsdfStructs[i].metallicRoughnessTextureID);
		printf("bsdfStructs[i].emissiveTextureID: %d\n", bsdfStructs[i].emissiveTextureID);

	}
}

void pathtraceInitBeforeMainLoop(SceneConfig * config) {

	// TODO: initialize any extra device memeory you need

	textureSize = hst_scene->textures.size();
	auto textureInfos = hst_scene->textures.data();
	cudaMalloc(&dev_textureInfos, (textureSize +1)* sizeof(TextureInfo)); // env_map also malloced
	cudaMemcpy(dev_textureInfos, textureInfos, textureSize * sizeof(TextureInfo), cudaMemcpyHostToDevice); // First copy textureSize of common texture

	cudaMalloc(&dev_textures, (textureSize +1)* sizeof(Texture)); // env_map also malloced
	checkCUDAError("cudaMalloc textures invalid!");
	unsigned char* dev_texture_data = nullptr;
	for (int i = 0; i < textureSize; i++)
	{
		cudaMalloc(&dev_texture_data, textureInfos[i].width * textureInfos[i].height * textureInfos[i].nrChannels * sizeof(unsigned char));
		cudaMemcpy(dev_texture_data, textureInfos[i].data.data(), textureInfos[i].width * textureInfos[i].height * textureInfos[i].nrChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
		initDeviceTextures << <1, 1 >> > (dev_textures[i], dev_textureInfos[i], dev_texture_data);
		printf("Loaded Texture %d\n", i);
		checkCUDAError("initDeviceTextures");
	}


	auto bsdfStructs = hst_scene->bsdfStructs.data();
	cudaMalloc(&dev_bsdfStructs, hst_scene->bsdfStructs.size() * sizeof(BSDFStruct));
	cudaMemcpy(dev_bsdfStructs, bsdfStructs, hst_scene->bsdfStructs.size() * sizeof(BSDFStruct), cudaMemcpyHostToDevice);

	initBSDFWithTextures << <1, 1 >> > (dev_bsdfStructs, dev_textures, hst_scene->bsdfStructs.size());
	checkCUDAError("initBSDFWithTextures");


	bvh = new BVHAccel();
	bvh->initBVH(hst_scene->triangles);
	auto triangles = bvh->orderedPrims.data();
	cudaMalloc(&hst_scene->dev_triangles, bvh->orderedPrims.size() * sizeof(Triangle));
	cudaMemcpy(hst_scene->dev_triangles, triangles, bvh->orderedPrims.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvhNodes, bvh->nodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhNodes, bvh->nodes.data(), bvh->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	hst_scene->initConfig(*config);
	hst_scene->initLights(bvh->orderedPrims);
	cudaMalloc(&dev_lights, hst_scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, hst_scene->lights.data(), hst_scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_lights");

	hst_scene->initEnvironmentalMap();
	const TextureInfo & env_map = hst_scene->config.env_map;
	cudaMemcpy(dev_textureInfos + textureSize, &env_map, 1 * sizeof(TextureInfo), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_texture_data, env_map.width * env_map.height * env_map.nrChannels * sizeof(unsigned char));
	cudaMemcpy(dev_texture_data, env_map.data.data(), env_map.width * env_map.height * env_map.nrChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_textureInfos1");
	initDeviceTextures << <1, 1 >> > (dev_textures[textureSize], dev_textureInfos[textureSize], dev_texture_data);

	checkCUDAError("cudaMemcpy dev_textureInfos");
	int triangle_size = bvh->orderedPrims.size();
	int blockSize = 256;
	dim3 initNormalTextureBlock((triangle_size + blockSize - 1) / blockSize);
	if (triangle_size) {
		initPrimitivesNormalTexture << <initNormalTextureBlock, blockSize >> > (hst_scene->dev_triangles, dev_textures, triangle_size);
	}
	else {
		printf("WARNING: NO TRIANGLES IN THE SCENE!\n");
	}
	
	checkCUDAError("pathtracer Init!");

}

void pathtraceInit(SceneConfig* hst_scene) {
	
	scene_config = hst_scene;

	const Camera& cam = scene_config->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first_iteration_first_bounce_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_iteration_first_bounce_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if USE_SORT_BY_MATERIAL
	int* dev_bsdfStructIDs_ptr;
	int * dev_bsdfStructIDs_copy_ptr;
	cudaMalloc(&(dev_bsdfStructIDs_ptr), pixelcount * sizeof(int));
	cudaMemset(dev_bsdfStructIDs_ptr, 0, pixelcount * sizeof(int));
	cudaMalloc(&dev_bsdfStructIDs_copy_ptr, pixelcount * sizeof(int));
	cudaMemset(dev_bsdfStructIDs_copy_ptr, 0, pixelcount * sizeof(int));
	dev_bsdfStructIDs = thrust::device_ptr<int>(dev_bsdfStructIDs_ptr);
	dev_bsdfStructIDs_copy = thrust::device_ptr<int>(dev_bsdfStructIDs_copy_ptr);
#endif // USE_SORT_BY_MATERIAL
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	//cudaFree(dev_triangles);
	//cudaFree(dev_bsdfStructs);
	//delete bvh;
	cudaFree(dev_first_iteration_first_bounce_cache_intersections);
#if USE_SORT_BY_MATERIAL
	cudaFree(dev_bsdfStructIDs.get());
	cudaFree(dev_bsdfStructIDs_copy.get());
#endif // SORT_BY_MATERIAL
	checkCUDAError("pathtraceFree");
}

void pathtraceFreeAfterMainLoop() {
	cudaFree(hst_scene->dev_triangles);
	cudaFree(dev_bsdfStructs);
	cudaFree(dev_textureInfos);

	/* TODO: Also need to free the texture that dev_texture point to*/
	cudaFree(dev_textures);
	delete bvh;
	delete hst_scene;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* hst_scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(const Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::random::uniform_real_distribution<float> u01(0,1);
		glm::vec2 jitter(u01(rng) - 0.5f, u01(rng) - 0.5f);
		//glm::vec2 jitter(u01(rng), u01(rng));
		// TODO: implement antialiasing by jittering the ray
		//glm::vec2 pixelCoord = glm::vec2(x, y) + jitter;
		//segment.ray.direction = glm::normalize(cam.view
		//	+ cam.right * cam.pixelLength.x * ((float)x + jitter[0] - (float)cam.resolution.x * 0.5f)
		//	+ cam.up * cam.pixelLength.y * ((float)y + jitter[1] - (float)cam.resolution.y * 0.5f)
		//);
		glm::vec3 pixelWorldCoord = cam.position + cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jitter[0] - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jitter[1] - (float)cam.resolution.y * 0.5f);
		auto rayDir = glm::normalize(pixelWorldCoord - cam.position);
		if (cam.aperture > 0.0f) {
			float lensU = u01(rng);
			float lensV = u01(rng);
			float lensRadius = cam.aperture;
			float focalDistance = cam.focalDistance;
			glm::vec3 focalPoint = cam.position + rayDir * focalDistance;
			glm::vec3 lensPoint = cam.position + cam.right * lensRadius * (lensU - 0.5f) + cam.up * lensRadius * (lensV - 0.5f);
			//printf("lensPoint: %f, %f, %f\n", lensPoint.x, lensPoint.y, lensPoint.z);
			segment.ray.origin = lensPoint;
			segment.ray.direction = glm::normalize(focalPoint - lensPoint);
			//segment.ray.direction = glm::normalize(pixelWorldCoord - cam.position);
		}
		else {
			segment.ray.origin = cam.position;
			segment.ray.direction = rayDir;
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.constantTerm = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.prevPDF = 0.0f;
	}
}

//TODO: Change to BVH in future!
__device__ bool intersectCore(BVHNode * nodes, int bvhNodes_size, Triangle * triangles, int triangles_size, Ray & ray, ShadeableIntersection& intersection) {
#if USE_BVH
	intersectBVH(nodes, bvhNodes_size, triangles, ray, intersection);
#else
	for (int i = 0; i < triangles_size; i++)
	{
		intersectTriangle(triangles[i], ray, &intersection);
	}
#endif
	return intersection.t > EPSILON;
}


__global__ void intersect(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Triangle* triangles
	, int triangles_size
	, ShadeableIntersection* intersections
	, BVHNode * bvhNodes
	, int bvhNodes_size
#if USE_SORT_BY_MATERIAL
	, thrust::device_ptr<int> bsdfStructIDs
	, thrust::device_ptr<int> bsdfStructIDs_copy
#endif
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		Ray & ray = pathSegments[path_index].ray;
		ray.min_t = 0.0;
		ray.max_t = FLT_MAX;
		intersections[path_index].t = -1;
		//intersectBVH(bvhNodes, bvhNodes_size, triangles, ray, intersections[path_index]);
		intersectCore(bvhNodes, bvhNodes_size, triangles, triangles_size, pathSegments[path_index].ray, intersections[path_index]);
#if USE_SORT_BY_MATERIAL
		bsdfStructIDs[path_index] = intersections[path_index].materialId;
		bsdfStructIDs_copy[path_index] = intersections[path_index].materialId;
#endif // SORT_BY_MATERIAL

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
	, int depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, BSDFStruct * bsdfStructs
	, Triangle * triangles
	, int triangles_size
	, BVHNode * bvhNodes
	, int bvhNodes_size
#if MIS
	, Light * lights
	, int lights_size
	, float inverse_sum_power
#endif
	, int screenHeight
	, int screenWidth
	, Texture * env_map
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);
			const float threshold_rr = 0.7f;
			float prob_rr = u01(rng);
			if (prob_rr > threshold_rr) {
				pathSegment.remainingBounces = 0;
				return;
			}
			const glm::vec3 rands(u01(rng), u01(rng), u01(rng));
			BSDFStruct bsdfStruct = bsdfStructs[intersection.materialId];
			initBSDF(bsdfStruct, intersection.uv);
			// If the material indicates that the object was a light, "light" the ray
			if (bsdfStruct.bsdfType == BSDFType::EMISSIVE) {
				pathSegments[idx].remainingBounces = 0;
				if (depth == 0) {
					pathSegment.color = bsdfStruct.emissiveFactor * bsdfStruct.strength;
				}
				else {
#if MIS
					float power_pdf = 1.0f / lights_size;
					float distance_sqr = intersection.t * intersection.t;
					float next_light_pdf = power_pdf * distance_sqr /
						(intersection.primitive->area() * abs(glm::dot(intersection.surfaceNormal, -pathSegment.ray.direction)));
					float weight = Math::PowerHeuristic(1, pathSegment.prevPDF, 1, next_light_pdf);
					pathSegment.color += weight * (bsdfStruct.emissiveFactor * bsdfStruct.strength) * pathSegment.constantTerm;
#else
					pathSegment.color += (bsdfStruct.emissiveFactor * bsdfStruct.strength) * pathSegment.constantTerm;
#endif
				}
			}
			else {
				glm::mat3x3 o2w;
				make_coord_space(o2w, intersection.surfaceNormal);
				glm::mat3x3 w2o(glm::transpose(o2w));
				if (bsdfStruct.normalTextureID != -1) {
					glm::vec3 sampledNormal = sampleTextureRGB(*(bsdfStruct.normalTexture), intersection.uv) * 2.0f - 1.0f;
					intersection.surfaceNormal = o2w * sampledNormal;
					make_coord_space(o2w, intersection.surfaceNormal);
					w2o = glm::transpose(o2w);
#pragma region bump-bug
					/* Fix bump mapping potential bug */
					//auto dir = -pathSegment.ray.direction;
					//glm::vec3 _wo = w2o * -pathSegment.ray.direction;
					//printf("_wo: %f, %f, %f dir: %f, %f, %f\n", _wo.x, _wo.y, _wo.z, dir.x, dir.y, dir.z);
					//pathSegment.color = glm::vec3(intersection.surfaceNormal);
					//return;
#pragma endregion
				}
				glm::vec3 intersect = pathSegment.ray.at(intersection.t);
				float pdf;
				glm::vec3 wo = w2o * -pathSegment.ray.direction;

#if  MIS
				if (lights_size > 0) {
					/* Uniformly pick a light, will change to selecting by importance in the future */
					thrust::uniform_int_distribution<int> uLight(0, lights_size - 1);
					int n_sample_light = 3;
					for (size_t i = 0; i < n_sample_light; i++)
					{
						int sampled_light_index = uLight(rng);
						const auto& light = lights[sampled_light_index];
						float light_pdf = 0.0f, scattering_pdf = 0.0f;
						const auto & ls = sampleLi(light, triangles[light.primIndex], intersection, glm::vec2(rands));
						if (ls.pdf > EPSILON) {
							light_pdf = ls.pdf;
							glm::vec3 light_wi = w2o * ls.wi;
							glm::vec3 f_direct_light = f(bsdfStruct, wo, light_wi, intersection.uv);
							scattering_pdf = PDF(bsdfStruct, wo, light_wi);
							if (!(f_direct_light.x < EPSILON && f_direct_light.y < EPSILON && f_direct_light.z < EPSILON)
								&& scattering_pdf > 0) {
								Ray light_ray;
								light_ray.direction = ls.wi;
								light_ray.origin = intersect;
								light_ray.min_t = EPSILON;
								light_ray.max_t = ls.distance - 1e-4f; // Do occulusion test by setting max_t
								ShadeableIntersection light_ray_intersection{ -1.0f }; // set t = -1
								if (!intersectCore(bvhNodes, bvhNodes_size, triangles, triangles_size, light_ray, light_ray_intersection)) {
									/* Determine whether delta */
									if (light.type== AREA_LIGHT) {
										float power_pdf = 1.0f / lights_size;
										// To do this we need to sample by power not uniformlly
										//float power_pdf = Math::luminance(Li) * triangles[light.primIndex].area() * TWO_PI * inverse_sum_power; 
										light_pdf = power_pdf * ls.pdf;
										float weight = Math::PowerHeuristic(1, light_pdf, 1, scattering_pdf);
										pathSegment.color += f_direct_light * ls.L * pathSegment.constantTerm * abs(light_wi.z) * weight / (light_pdf * n_sample_light * threshold_rr);
									}
								}

							}
						}

					}
				}
#endif			
				glm::vec3 wi;
				glm::vec3 bsdf = sample_f(bsdfStruct, wo, wi, &pdf, rng, intersection.uv, rands);
				if (glm::length(wi) < 1.0f - EPSILON) {
					pathSegment.remainingBounces = 0;
				}
				else {
					pathSegment.constantTerm *= (bsdf * abs(wi.z) / (pdf * threshold_rr));
					pathSegment.ray.direction = o2w * wi;
					pathSegment.ray.origin = intersect;
					pathSegment.remainingBounces--;
					pathSegment.prevPDF = pdf;
				}
			}
		}
		else {
			//const auto & d = pathSegment.ray.direction;
			//const float phi = atan2f(d.z, d.x);
			//const float theta = acosf(d.y);
			//const float u = (phi + PI) / (2 * PI);
			//const float v = theta / PI;
			//const glm::vec2 uv(u, v);
			//const float env_map_strength = 3.0f;
			//auto env_map_sample = sampleTextureRGB(*env_map, uv) * env_map_strength;
			//if (depth == 0) {
			//	pathSegment.color += env_map_sample * pathSegment.constantTerm;
			//}
			//else {
			//	pathSegment.color += env_map_sample * pathSegment.constantTerm;
			//}
			pathSegments[idx].remainingBounces = 0;
			//printf("pathSegment.color: %f %f %f\n", pathSegment.color.x, pathSegment.color.y, pathSegment.color.z);

		}
		
		if (glm::all(glm::isinf(pathSegment.color)) || glm::all(glm::isnan(pathSegment.color))) {
			assert(0);
			pathSegment.color = glm::vec3(0.0f);
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

__global__ void sampleScreen(glm::vec3 * dev_image, Texture & texture, int imageWidth, int imageHeight) {
	for (size_t x = 0; x < imageWidth; x++)
	{
		for (size_t y = 0; y < imageHeight; y++)
		{
			auto sampleResult = sampleTextureRGBA(texture, glm::vec2(float(x) / imageWidth, float(y) / imageHeight));
			dev_image[x + y * imageWidth] = glm::vec3(sampleResult.x, sampleResult.y, sampleResult.z);
		}
	}
}

struct HasHit{
    __host__ __device__ bool operator()(const PathSegment & path) const {
        return path.remainingBounces != 0;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	checkCUDAError("before generate camera ray");

	const int traceDepth = scene_config->state.traceDepth;
	const Camera& cam = scene_config->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into hst_scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {


		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if USE_FIRST_BOUNCE_CACHE
		if (iter == 1) {
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


			intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_triangles
				, scene->triangles.size()
				, dev_intersections
				, dev_bvhNodes
				, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
				, dev_bsdfStructIDs
				, dev_bsdfStructIDs_copy
#endif
				);
			if (depth == 0) {
				cudaMemcpy(dev_first_iteration_first_bounce_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			//hasFirstBounceCache = true;
		}
		else {
			if (depth == 0) {
				cudaMemcpy(dev_intersections, dev_first_iteration_first_bounce_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_triangles
					, scene->triangles.size()
					, dev_intersections
					, dev_bvhNodes
					, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
					, dev_bsdfStructIDs
					, dev_bsdfStructIDs_copy
#endif
					);
			}
		}
#else
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


		intersect << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, hst_scene->dev_triangles
			, hst_scene->triangles.size()
			, dev_intersections
			, dev_bvhNodes
			, bvh->nodes.size()
#if USE_SORT_BY_MATERIAL
			, dev_bsdfStructIDs
			, dev_bsdfStructIDs_copy
#endif
			);
#endif

#if USE_SORT_BY_MATERIAL
		thrust::sort_by_key(thrust::device, dev_bsdfStructIDs, dev_bsdfStructIDs + num_paths, dev_paths);
		thrust::sort_by_key(thrust::device, dev_bsdfStructIDs_copy, dev_bsdfStructIDs_copy + num_paths, dev_intersections);
#endif // USE_SORT_BY_MATERIAL

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

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
			depth,
			dev_intersections,
			dev_paths,
			dev_bsdfStructs,
			hst_scene->dev_triangles,
			hst_scene->triangles.size(),
			dev_bvhNodes,
			bvh->nodes.size()
#if MIS
			, dev_lights
			, hst_scene->lights.size()
			, hst_scene->inverse_sum_power
#endif
			, cam.resolution.y
			, cam.resolution.x
			, dev_textures + textureSize
			);
		cudaDeviceSynchronize();
		checkCUDAError("shadeMaterial failed\n");

		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, HasHit());
		num_paths = dev_path_end - dev_paths;

		iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.
		//iterationComplete = (true);
		depth++;
		
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
	//int imageWidth = scene_config->state.camera.resolution.x;
	//int imageHeight = scene_config->state.camera.resolution.y;
	//sampleScreen << <1, 1 >> > (dev_image, dev_textures[0], imageWidth, imageHeight);

	// Test texture sampling
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);



	// Retrieve image from GPU
	cudaMemcpy(scene_config->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
