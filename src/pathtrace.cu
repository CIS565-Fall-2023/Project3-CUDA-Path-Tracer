#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/complex.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "BVH.h"
//for testing
#include <bitset>

#define ERRORCHECK 1
#define DENOISING 0
#define SVGF 0
#define ANTIALISING 0
#define CACHEFIRSTRAY 1
#define DEFOCUSING 0
#define BVHON 1
#define DIRECTLIGHTING 1

#define LEN 7.0f

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
static glm::vec3* dev_postImage = NULL;
static Geom* dev_geoms = NULL;
static Geom* dev_lightSources = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static glm::vec3* dev_textures = NULL;
static AABB* dev_bboxes = NULL;
static Texture* dev_texes = NULL;
static int* dev_offsets = NULL;
static ShadeableIntersection* dev_cachedintersections = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_BVHNodes = NULL;
// For denoising
static glm::vec3* dev_normalBuffer;
static glm::vec3* dev_albedoBuffer;
static float* dev_zBuffer;
static int* dev_objIndex;

int lightSrcNumber;

struct stay_bouncing
{
	__host__ __device__
		bool operator()(const PathSegment path)
	{
		return path.remainingBounces != 0;
	}
};

__host__ __device__ bool operator<(const ShadeableIntersection& inter1, const ShadeableIntersection& inter2)
{
	return inter1.materialId < inter2.materialId;
}

void initBufferInfo(int pixels)
{
	cudaMalloc(&dev_normalBuffer, pixels * sizeof(glm::vec3));
	cudaMemset(dev_normalBuffer, 0, pixels * sizeof(glm::vec3));
	cudaMalloc(&dev_albedoBuffer, pixels * sizeof(glm::vec3));
	cudaMemset(dev_albedoBuffer, 0, pixels * sizeof(glm::vec3));
	cudaMalloc(&dev_zBuffer, pixels * sizeof(float));
	cudaMemset(dev_zBuffer, 0, pixels * sizeof(float));
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void iniLightInfo(Scene* scene)
{
	//get the needed light infomation
	lightSrcNumber = 0;
	std::vector<Geom> lightSources;
	for (int i = 0; i < hst_scene->geoms.size(); i++)
	{
		Geom geo = (hst_scene->geoms)[i];
		if ((hst_scene->materials)[geo.materialid].emittance > 0)
		{
			++lightSrcNumber;
			lightSources.push_back(geo);
		}
	}
	cudaMalloc(&dev_lightSources, lightSrcNumber * sizeof(Geom));
	cudaMemcpy(dev_lightSources, lightSources.data(), lightSrcNumber * sizeof(Geom), cudaMemcpyHostToDevice);
	lightSources.clear();
}

void initTexInfo(Scene* scene)
{
	std::vector<int> texoffsets;

	cudaMalloc(&dev_textures, scene->texData.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_textures, scene->texData.data(), scene->texData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_texes, scene->textures.size() * sizeof(Texture));
	cudaMemcpy(dev_texes, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
}

void initMeshInfo(Scene* scene)
{
	hst_scene = scene;
	std::vector<int> offsets;
	std::vector<AABB> bboxes;
	std::vector<Triangle> triangles;
	offsets.push_back(0);
	int meshCount = scene->gltfMeshes.size();
	for (int i = 0; i < meshCount; i++)
	{
		const GLTFMesh& mesh1 = (hst_scene->gltfMeshes)[i];
		copy(mesh1.triangles.begin(), mesh1.triangles.end(), back_inserter(triangles));
		// for uv, it is 2/3 of the offset
		offsets.push_back(triangles.size());
		bboxes.push_back(AABB{ mesh1.bbmin, mesh1.bbmax });
	}
	cudaMalloc(&dev_triangles, triangles.size()*sizeof(Triangle));
	cudaMemcpy(dev_triangles, triangles.data() ,triangles.size()*sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_bboxes, bboxes.size() * sizeof(AABB));
	cudaMemcpy(dev_bboxes, bboxes.data(), bboxes.size() * sizeof(AABB), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_offsets, offsets.size() * sizeof(int));
	cudaMemcpy(dev_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_BVHNodes, (triangles.size() * 2 - meshCount) * sizeof(BVHNode));
	cudaMemset(dev_BVHNodes, 0, (triangles.size() * 2 - meshCount) * sizeof(BVHNode));
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_postImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_postImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	initMeshInfo(scene);
	initTexInfo(scene);
	iniLightInfo(scene);
	initBufferInfo(pixelcount);
#if CACHEFIRSTRAY
	cudaMalloc(&dev_cachedintersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cachedintersections, 0, pixelcount * sizeof(ShadeableIntersection));
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
	cudaFree(dev_bboxes);
	cudaFree(dev_offsets);
	cudaFree(dev_textures);
	cudaFree(dev_texes);
	cudaFree(dev_lightSources);
	cudaFree(dev_postImage);
	cudaFree(dev_normalBuffer);
	cudaFree(dev_albedoBuffer);
	cudaFree(dev_zBuffer);
	cudaFree(dev_triangles);
	cudaFree(dev_BVHNodes);
#if CACHEFIRSTRAY
	cudaFree(dev_cachedintersections);
#endif
	checkCUDAError("pathtraceFree");
}

void updateBBox(Scene* scene)
{
	std::vector<AABB> bboxes;
	for (int i = 0; i < scene->gltfMeshes.size(); i++)
	{
		const GLTFMesh& mesh1 = (hst_scene->gltfMeshes)[i];
		bboxes.push_back(AABB{ mesh1.bbmin, mesh1.bbmax });
	}
	cudaMemcpy(dev_bboxes, bboxes.data(), bboxes.size() * sizeof(AABB), cudaMemcpyHostToDevice);
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
		segment.currentNormal = glm::vec3(0.0f, 0.0f, 0.0f);

#if !CACHEFIRSTRAY && ANTIALIASING
		// TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rngX = makeSeededRandomEngine(iter, x, index);
		thrust::default_random_engine rngY = makeSeededRandomEngine(iter, y, index);
		thrust::uniform_real_distribution<float> u11(-0.5f, 0.5f);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u11(rngX))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u11(rngY))
		);
#else
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
#if DEFOCUSING
		thrust::default_random_engine rngX = makeSeededRandomEngine(iter, x, index);
		thrust::default_random_engine rngY = makeSeededRandomEngine(iter, y, index);
		/*
		thrust::uniform_real_distribution<float> u360(0.0f, 360.0f);
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
		float degree = u360(rngX);
		float len = u01(rngY);
		glm::vec3 defocus = glm::vec3(cos(degree) * len, sin(degree) * len, 0);*/
		thrust::uniform_real_distribution<float> u00(-0.7f, 0.7f);
		glm::vec3 defocus = glm::vec3(u00(rngX), u00(rngY), 0);
		segment.ray.origin += defocus;

		glm::vec3 focusPoint = LEN * segment.ray.direction;
		glm::vec3 tmpDir = focusPoint - defocus;
		segment.ray.direction = glm::normalize(tmpDir);
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
	, int* meshOffsets
	, AABB* bboxes
	, Triangle* meshTriangles
	, BVHNode* nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv = glm::vec2(FLT_MAX, FLT_MAX);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

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
			else if (geom.type == MESH)
			{
				int meshInd = geom.meshid;
#if !BVHON
				t = meshIntersectionTest(geom, meshTriangles, meshOffsets[meshInd],
					meshOffsets[meshInd + 1], bboxes[meshInd],
					pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
#else
				t = traverseTree(nodes, geom, meshTriangles, meshOffsets[meshInd],
					meshOffsets[meshInd + 1], bboxes[meshInd], pathSegment.ray,
					tmp_intersect, tmp_normal, tmp_uv, outside, meshInd);
#endif
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
				uv = tmp_uv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			intersections[path_index].hasUV = false;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].textureId = geoms[hit_geom_index].textureid;
			intersections[path_index].surfaceNormal = normal;
			if (geoms[hit_geom_index].type != MESH)
			{
				intersections[path_index].hasUV = false;
				intersections[path_index].interUV = glm::vec2(FLT_MAX, FLT_MAX);
			}
			else
			{
				intersections[path_index].hasUV = true;
				intersections[path_index].interUV = uv;
			}
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
	, Texture* texes
	, glm::vec3* textures
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
			glm::vec3 c = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= c * material.emittance;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				int texID = intersection.textureId;
				glm::vec2 uv = intersection.interUV;
				if (texID > -1 && intersection.hasUV == true)
				{
					Texture tmpTex = texes[texID];
					int i = tmpTex.width * intersection.interUV[0] - 0.5f;
					int j = tmpTex.height * intersection.interUV[1] - 0.5f;
					int colorIndex = j * tmpTex.width + i + tmpTex.start;
					c = textures[colorIndex];
				}
				pathSegments[idx].color *= (c * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * c) * 0.7f;
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

__device__ glm::vec3 getLightDir(glm::vec3 origin, Geom lightSource)
{
	glm::vec3 sourcePoint;
	thrust::default_random_engine xrng = makeSeededRandomEngine((int)origin.x, (int)(origin.y * lightSource.meshid), (int)(origin.z * lightSource.textureid));
	thrust::default_random_engine yrng = makeSeededRandomEngine((int)(origin.x * lightSource.materialid), (int)origin.y, (int)(origin.z * lightSource.textureid));
	thrust::default_random_engine zrng = makeSeededRandomEngine((int)(origin.x * lightSource.materialid), (int)(origin.y * lightSource.meshid), (int)origin.z);
	thrust::uniform_real_distribution<float> u11(-1, 1);
	if (lightSource.type == CUBE)
	{
		 sourcePoint = (glm::vec3)(lightSource.transform * glm::vec4(u11(xrng) * 0.5f, u11(yrng) * 0.5f, u11(zrng) * 0.5f, 1.0f));
	}
	else
	{
		sourcePoint = (glm::vec3)(lightSource.transform * glm::vec4(u11(xrng) * 0.35f, u11(yrng) * 0.35f, u11(zrng) *0.35f, 1.0f));
	}

	return glm::normalize(sourcePoint - origin);
}

__global__ void initBufferData(
	int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Texture* textures
	, glm::vec3* texdata
	, glm::vec3* albedoBuffer
	, glm::vec3* normalBuffer
	, float* zBuffer
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		glm::vec3 tmpColor;

		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			Material material = materials[intersection.materialId];
			glm::vec3 interPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
			normalBuffer[idx] = intersection.surfaceNormal;
			zBuffer[idx] = interPoint.z;
			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				albedoBuffer[idx] = material.color * material.emittance;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				int texID = intersection.textureId;
				glm::vec2 uv = intersection.interUV;
				if (texID > -1 && intersection.hasUV == true)
				{
					Texture tmpTex = textures[texID];
					int i = tmpTex.width * intersection.interUV[0] - 0.5f;
					int j = tmpTex.height * intersection.interUV[1] - 0.5f;
					int colorIndex = j * tmpTex.width + i + tmpTex.start;
					albedoBuffer[idx] = texdata[colorIndex] * material.color;
				}
				else
				{
					albedoBuffer[idx] = material.color;
				}
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {

			albedoBuffer[idx] = glm::vec3(0.0f);
			normalBuffer[idx] = glm::vec3(0.0f, 0.0f, 0.0f);
			zBuffer[idx] = -FLT_MAX;
		}
	}
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Texture* textures
	, glm::vec3* texdata
	, Geom* lightSource
	, int lightSrcNumber
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

			glm::vec3 c;

			Material material = materials[intersection.materialId];
			c = material.color;


			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= c * material.emittance;
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				pathSegments[idx].color *= u01(rng); // apply some noise because why not*/
				glm::vec3 interPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

				int texID = intersection.textureId;
				glm::vec2 uv = intersection.interUV;
				if (texID > -1 && intersection.hasUV == true)
				{
					Texture tmpTex = textures[texID];
					int i = tmpTex.width * intersection.interUV[0] - 0.5f;
					int j = tmpTex.height * intersection.interUV[1] - 0.5f;
					int colorIndex = j * tmpTex.width + i + tmpTex.start;
					c = texdata[colorIndex];
				}
				material.color = c * material.color;
				scatterRay(pathSegments[idx], interPoint, intersection.surfaceNormal, material, rng);
#if DIRECTLIGHTING 1
				if (pathSegments[idx].remainingBounces == 1)
				{
					thrust::uniform_real_distribution<float> u01(0, 1);
					thrust::uniform_real_distribution<int> uLight(0, lightSrcNumber - 1);
					glm::vec3 directLightDir = getLightDir(pathSegments[idx].ray.origin, lightSource[uLight(rng)]);
					pathSegments[idx].ray.direction = glm::normalize(directLightDir);
				}
# endif
				if (pathSegments[idx].remainingBounces == 0)
				{
					pathSegments[idx].color = glm::vec3(0.0f);
				}
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
}

//Extremely Blur!
__global__ void naiveFilter(int nPaths, glm::vec3* image, glm::vec3* postImage, int width, int height)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < nPaths)
	{
		int radius = 1;
		//a 3x3 filter
		int y = index / width;
		int x = index - width * y;
		int xStart = thrust::max(0, x - 1);
		int xEnd = thrust::min(x + 1, width - 1);
		int yStart = thrust::max(0, y - 1);
		int yEnd = thrust::min(y + 1, height - 1);

		float totalWeight = 0.0f;
		glm::vec3 weights = glm::vec3(4.0f, 2.0f, 1.0f);
		glm::vec3 weightedColor = glm::vec3(0.0f, 0.0f, 0.0f);
		for (int i = yStart; i < yEnd + 1; i++)
		{
			for (int j = xStart; j < xEnd + 1; j++)
			{
				int diff = abs(i - y) + abs(j - x);
				weightedColor += image[i * width + j] * weights[diff];
				totalWeight += weights[diff];
			}
		}
		postImage[index] = weightedColor / totalWeight;
	}
}

//Extremely Blur and Looks Fake!
__global__ void jointBilateralFilter(int nPaths, glm::vec3* image, glm::vec3* postImage, int width, int height, 
	glm::vec3* normalBuffer, glm::vec3* albedoBuffer)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < nPaths)
	{
		int radius = 7;
		int y = index / width;
		int x = index - width * y;
		int xStart = thrust::max(0, x - radius);
		int xEnd = thrust::min(x + radius, width - 1);
		int yStart = thrust::max(0, y - radius);
		int yEnd = thrust::min(y + radius, height - 1);
		float g = 0;
		float totalWeight = 0.0f;
		glm::vec3 weightedColor = glm::vec3(0.0f, 0.0f, 0.0f);
		for (int i = yStart; i < yEnd + 1; i++)
		{
			for (int j = xStart; j < xEnd + 1; j++)
			{
				glm::vec2 indexDiff = (glm::vec2(i, j) - glm::vec2(y, x)) * 0.143f;
				glm::vec3 normalDiff = normalBuffer[i * width + j] - normalBuffer[index];
				glm::vec3 albedoDiff = albedoBuffer[i * width + j] - albedoBuffer[index];
				float sqIndex = glm::dot(indexDiff, indexDiff);
				float sqNormal = glm::dot(normalDiff, normalDiff);
				float sqAlbedo = glm::dot(albedoDiff, albedoDiff);
				float weight = exp(-sqIndex / 1.0f - sqNormal / 1.0f - sqAlbedo / 1.0f);
				weightedColor += image[i * width + j] * weight;
				totalWeight += weight;
			}
		}
		postImage[index] = weightedColor / totalWeight;
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
	const int traceDepth = hst_scene->state.traceDepth;
	//const int traceDepth = 1;
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
	int tempPaths = num_paths;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		dim3 numblocksPathSegmentTracing = (tempPaths + blockSize1d - 1) / blockSize1d;
		if (iter == 1)
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, tempPaths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_offsets
				, dev_bboxes
				, dev_triangles
				, dev_BVHNodes
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			if (depth == 0)
			{
				//write to cache
				cudaMemcpy(dev_cachedintersections, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				initBufferData << <numblocksPathSegmentTracing, blockSize1d >> > (
					tempPaths
					, dev_intersections
					, dev_paths
					, dev_materials
					, dev_texes
					, dev_textures
					, dev_albedoBuffer
					, dev_normalBuffer
					, dev_zBuffer
				);
			}
		}
		else
		{
			if (depth == 0)
			{
				//read from cache
				cudaMemcpy(dev_intersections, dev_cachedintersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else
			{
				// tracing
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, tempPaths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, dev_offsets
					, dev_bboxes
					, dev_triangles
					, dev_BVHNodes
					);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();

			}
		}
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + tempPaths, dev_paths);
		depth++;
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.


		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			tempPaths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_texes,
			dev_textures,
			dev_lightSources,
			lightSrcNumber
			);
		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, stay_bouncing());
		//std::cout << tempPaths << std::endl;
		tempPaths = dev_path_end - dev_paths;
		//std::cout << tempPaths << std::endl;
		iterationComplete = (tempPaths == 0); // TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
#if DENOISING
	int width = cam.resolution.x;
	int height = cam.resolution.y;
#if SVGF
	jointBilateralFilter << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_postImage, width, height,
	dev_normalBuffer, dev_albedoBuffer);
#else
	naiveFilter << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_postImage, width, height);
#endif
	std::swap(dev_image, dev_postImage);
#endif
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

void buildBVHTree(int startIndexBVH, int startIndexTri, GLTFMesh mesh1, int triCount)
{
	const int blockSize1d = 128;
	unsigned int* dev_mortonCodes = NULL;
	cudaMalloc(&dev_mortonCodes, triCount * sizeof(unsigned int));
	cudaMemset(dev_mortonCodes, 0, triCount * sizeof(unsigned int));
	unsigned char* dev_ready = NULL;
	cudaMalloc(&dev_ready, (triCount * 2 - 1) * sizeof(unsigned char));
	cudaMemset(dev_ready, 0, triCount * sizeof(unsigned char));
	cudaMemset(&dev_ready[triCount - 1], 1, triCount * sizeof(unsigned char));

	static BVHNode* dev_tmpBVHNodes = NULL;
	cudaMalloc(&dev_tmpBVHNodes, (triCount * 2 - 1) * sizeof(BVHNode));
	cudaMemset(dev_tmpBVHNodes, 0, (triCount * 2 - 1) * sizeof(BVHNode));

	dim3 numblocks = (triCount + blockSize1d - 1) / blockSize1d;
	buildLeafMorton << <numblocks, blockSize1d >> > (startIndexTri, triCount, mesh1.bbmin.x, mesh1.bbmin.y, mesh1.bbmin.z, mesh1.bbmax.x, mesh1.bbmax.y, mesh1.bbmax.z,
		dev_triangles, dev_tmpBVHNodes, dev_mortonCodes);
	
	thrust::stable_sort_by_key(thrust::device, dev_mortonCodes, dev_mortonCodes + triCount, dev_tmpBVHNodes + triCount - 1);


	/*
	unsigned int* hstMorton = (unsigned int*)malloc(sizeof(unsigned int) * triCount);
	cudaMemcpy(hstMorton, dev_mortonCodes, triCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 20; i++)
	{
		cout << std::bitset<30>(hstMorton[i]) << endl;
	}
	cout << endl;
	free(hstMorton);*/
	
	

	buildSplitList << <numblocks, blockSize1d >> > (triCount, dev_mortonCodes, dev_tmpBVHNodes);

	//can use atomic operation for further optimization
	for (int i = 0; i < triCount; i++)
	{
		buildBBoxes << <numblocks, blockSize1d >> > (triCount, dev_tmpBVHNodes, dev_ready);
	}

	cudaMemcpy(dev_BVHNodes + startIndexBVH, dev_tmpBVHNodes, (triCount * 2 - 1) * sizeof(BVHNode), cudaMemcpyDeviceToDevice);
	
	/*
	BVHNode* hstBVHNodes = (BVHNode*)malloc(sizeof(BVHNode) * (startIndexBVH + 2 * triCount - 1));
	cudaMemcpy(hstBVHNodes, dev_BVHNodes, sizeof(BVHNode) * (startIndexBVH + 2 * triCount - 1), cudaMemcpyDeviceToHost);
	for (int i = 0; i < startIndexBVH + 2 * triCount - 1; i++)
	{
		cout << i << ": " << hstBVHNodes[i].leftIndex << "," << hstBVHNodes[i].rightIndex << "  parent:" << hstBVHNodes[i].parent << endl;
		cout << i << ": " << hstBVHNodes[i].bbox.max.x << "," << hstBVHNodes[i].bbox.max.y << "," << hstBVHNodes[i].bbox.max.z << endl;
		//cout << i << ": " << hstBVHNodes[i].bbox.min.x << "," << hstBVHNodes[i].bbox.min.y << "," << hstBVHNodes[i].bbox.min.z << endl;
	}
	cout << endl;
	cout << endl;
	cout << endl;
	free(hstBVHNodes);*/
	

	cudaFree(dev_ready);
	cudaFree(dev_mortonCodes);
	cudaFree(dev_tmpBVHNodes);
}