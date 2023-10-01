#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <numeric>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

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
static PathSegment* dev_tempPaths = NULL;
static PathSegment* dev_pathsBuffer = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_tempIntersections = NULL;
static ShadeableIntersection* dev_intersectionsBuffer = NULL;
static int* dev_bools = NULL;
static int* dev_nbools = NULL;
static int* dev_scanBools = NULL;
static int* dev_scanNBools = NULL;
static Triangle* dev_tris = NULL;
static BoundingBox* dev_bvh = NULL;
static TriangleArray* dev_triArr = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void printArr(int n, int* odata, int* dev_odata) {
	cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy dev_odata -> odata for printArr failed!");
	for (int i = 0; i <= n / 10; i++) {
		for (int j = 0; j < 10 && j < n - 10 * i; j++) {
			std::cout << odata[i * 10 + j] << "  ";
		}std::cout << std::endl;
	}std::cout << std::endl << std::endl;
}

void printArr(int begin, int n, int* dev_odata) {
	int o[10];
	for (int i = 0; i <= n / 10; i++) {
		cudaMemcpy(o, dev_odata + begin + 10 * i, sizeof(int) * 10, cudaMemcpyDeviceToHost);
		if (o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]+o[8]+o[9] == -10) {
			continue;
		}
		checkCUDAError("cudaMemcpy dev_odata -> odata for printArr failed!");
		for (int j = 0; j < 10 && j < n - 10 * i; j++) {
			std::cout << o[j] << "  ";
		}std::cout << std::endl;
	}std::cout << std::endl << std::endl;
}

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
	cudaMalloc(&dev_tempPaths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_tempIntersections, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_bools, pixelcount * sizeof(int));
	cudaMalloc(&dev_nbools, pixelcount * sizeof(int));
	cudaMalloc(&dev_scanBools, pixelcount * sizeof(int));
	cudaMalloc(&dev_scanNBools, pixelcount * sizeof(int));

#ifdef CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_intersectionsBuffer, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_pathsBuffer, pixelcount * sizeof(PathSegment));
#endif

#ifdef USING_BVH
	cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(BoundingBox));
	cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(BoundingBox), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_triArr, scene->triArr.size() * sizeof(TriangleArray));
	cudaMemcpy(dev_triArr, scene->triArr.data(), scene->triArr.size() * sizeof(TriangleArray), cudaMemcpyHostToDevice);
#endif

	checkCUDAError("pathtraceInit");	
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	cudaFree(dev_bools);
	cudaFree(dev_nbools);
	cudaFree(dev_scanBools);
	cudaFree(dev_scanNBools);
	cudaFree(dev_tempPaths);
	cudaFree(dev_tempIntersections);
#ifdef CACHE_FIRST_BOUNCE
	cudaFree(dev_intersectionsBuffer);
	cudaFree(dev_pathsBuffer);
#endif
#ifdef USING_BVH
	cudaFree(dev_bvh);
	cudaFree(dev_triArr);
#endif
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

#ifdef JITTER_RAY
		thrust::random::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
		thrust::uniform_real_distribution<float> u1(-1, 1);
		thrust::uniform_real_distribution<float> u02PI(0, TWO_PI);
		float rz = sqrt(u1(rng));
		rz = ((rz > 0) ? 1 : -1) * sqrt(abs(rz));
#endif

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
#ifdef JITTER_RAY
			+ glm::normalize(glm::vec3(__sinf(u02PI(rng)), __cosf(u02PI(rng)), rz)) * 1e-3f
#endif
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.refractionBefore = false;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersectionsNaive(
	int depth
	, int num_paths
	, const PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Triangle* tris
	, int tris_size
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
		int hit_index = -1;
		bool hit_geom = true;
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
				hit_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		for (int i = 0; i < tris_size; i++) {
			Triangle& tri = tris[i];
			t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_index = i;
				hit_geom = false;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_geom ? geoms[hit_index].materialid : tris[hit_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}


__device__ __host__ void searchTriArrIntersect(
	Ray& ray, float& t_min, int& hit_index, TriangleArray& triIndices, Triangle* tris,
	glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside
) {
	#pragma unroll
	for (int j = 0; j < BBOX_TRI_NUM; j++) {
		int ti = triIndices.triIds[j];
		if (ti < 0) { return; }
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		float t = triangleIntersectionTest(tris[ti], ray, tmp_intersect, tmp_normal, outside);
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_index = ti;
			intersectionPoint = tmp_intersect;
			normal = tmp_normal;
		}
	}
}

__device__  float bvhSearch(
	const Ray& ray, int& hit_index
	, Triangle* tris, int tris_size
	, BoundingBox* bvh, int bvh_size
	, TriangleArray* tri_arr, int tri_arr_size
	, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside
	, int blockDim, int threadId
) {

	float t_min = 1e4;
	// float bt_min = 1e4;

	
	//for (int i = 0; i < bvh_size; i++) {  ///// Test BVH
	//	BoundingBox& bbox = bvh[i]; // if (boundingboxIntersectionTest(ray, bbox.min, bbox.max))
	//	// reach leaf node of bvh
	//	if (bbox.triArrId >= 0) {
	//		searchTriArrIntersect(ray, t_min, hit_index, tri_arr[bbox.triArrId], tris, intersectionPoint, normal, outside);
	//	}
	//}
	

	// __shared__ int arr[BLOCK_SIZE_1D][BVH_GPU_STACK_SIZE];
	int arr[BVH_GPU_STACK_SIZE];

	// int arr[BVH_GPU_STACK_SIZE];
	int sign = 0;
	arr[0] = 0;

	while (sign >= 0) {
		BoundingBox& bbox = bvh[arr[sign]];
		sign--;
		glm::vec3 minDiff = bbox.min - ray.origin;
		glm::vec3 maxDiff = bbox.max - ray.origin;
		bool inside = (minDiff.x * maxDiff.x <= 0 &&
			minDiff.y * maxDiff.y <= 0 &&
			minDiff.z * maxDiff.z <= 0);
		// bool inside = false;

		float bt = (abs(ray.direction[0]) < 1e-5) ? -1.0f : max(-1.0f, min(minDiff.x / ray.direction[0], maxDiff.x / ray.direction[0]));
		bt = (abs(ray.direction[1]) < 1e-5) ? bt : max(bt, min(minDiff.y / ray.direction[1], maxDiff.y / ray.direction[1]));
		bt = (abs(ray.direction[2]) < 1e-5) ? bt : max(bt, min(minDiff.z / ray.direction[2], maxDiff.z / ray.direction[2]));

		if (inside || (bt > -1e-3 && bt < t_min))
		{
			// reach leaf node of bvh
			if (bbox.triArrId >= 0) {
				// searchTriArrIntersect(ray, t_min, hit_index, tri_arr[bbox.triArrId], tris, intersectionPoint, normal, outside);
				
				TriangleArray& triIndices = tri_arr[bbox.triArrId];
				// #pragma unroll
				for (int j = 0; j < BBOX_TRI_NUM; j++) {
					int ti = triIndices.triIds[j];
					if (ti < 0) { break; }
					glm::vec3 tmp_intersect;
					glm::vec3 tmp_normal;
					float t = triangleIntersectionTest(tris[ti], ray, tmp_intersect, tmp_normal, outside);
					if (t > 0.0f && t_min > t)
					{
						//if (t_min < bt) {
						//	/////
						//}
						//if (t < bt) {
						//	/////
						//	float ori[3] = { ray.origin.x, ray.origin.y, ray.origin.z };
						//	float dir[3] = { ray.direction.x, ray.direction.y, ray.direction.z };
						//	int bboxId = arr[threadId][sign + 1];
						//	int a = 1;
						//}
						t_min = t;
						hit_index = ti;
						intersectionPoint = tmp_intersect;
						normal = tmp_normal;
					}
				}
			}
			// keep searching
			else if (sign + 2 < BVH_GPU_STACK_SIZE) {
				arr[sign + 1] = bbox.leftId;
				arr[sign + 2] = bbox.rightId;
				sign += 2;
			}
			else {
				return -99.0f;
			}
		}
	}
	
	return hit_index < 0 ? -1 : t_min;

}


__global__ void computeIntersectionsBVH(
	int depth
	, int num_paths
	, const PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Triangle* tris
	, int tris_size
	, BoundingBox* bvh
	, int bvh_size
	, TriangleArray* tri_arr
	, int tri_arr_size
	, ShadeableIntersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = 1e5;
		int hit_index = -1;
		bool hit_geom = true;
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
				hit_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}


		//t = bvhSearch(pathSegment.ray, hit_tri_index,
		//	tris, tris_size, bvh, bvh_size, tri_arr, tri_arr_size, tmp_intersect, tmp_normal, outside,
		//	blockDim.x, threadIdx.x);

		// float t_min = 1e4;

		// __shared__ int arr[BLOCK_SIZE_1D][BVH_GPU_STACK_SIZE];
		int arr[BVH_GPU_STACK_SIZE];

		// int arr[BVH_GPU_STACK_SIZE];
		int sign = 0;
		arr[0] = 0;

		float o[3] = { pathSegment.ray.origin.x, pathSegment.ray.origin.y, pathSegment.ray.origin.z };
		float dir[3] = { pathSegment.ray.direction.x, pathSegment.ray.direction.y, pathSegment.ray.direction.z };
		float inv_dir[3] = {
			(abs(dir[0]) < 1e-5) ? 0.0f : 1.0f / dir[0],
			(abs(dir[1]) < 1e-5) ? 0.0f : 1.0f / dir[1],
			(abs(dir[2]) < 1e-5) ? 0.0f : 1.0f / dir[2]
		};
		// bool dir_zero[3] = { (abs(dir[0]) < 1e-5), (abs(dir[1]) < 1e-5), (abs(dir[2]) < 1e-5) };

		while (sign >= 0) {
			BoundingBox& bbox = bvh[arr[sign]];
			int taid = bbox.triArrId;
			glm::vec3& bmin = bbox.min;
			glm::vec3& bmax = bbox.max;
			sign--;

			// glm::vec3 minDiff = bbox.min - pathSegment.ray.origin;
			// glm::vec3 maxDiff = bbox.max - pathSegment.ray.origin;
			float minDiff[3] = { bmin[0] - o[0], bmin[1] - o[1], bmin[2] - o[2] };
			float maxDiff[3] = { bmax[0] - o[0], bmax[1] - o[1], bmax[2] - o[2] };
			//bool inside = (minDiff[0] * maxDiff[0] <= 0 &&
			//	minDiff[1] * maxDiff[1] <= 0 &&
			//	minDiff[2] * maxDiff[2] <= 0);
			// bool inside = false;


			float bt = max(max(
				min(minDiff[1] * inv_dir[1], maxDiff[1] * inv_dir[1]),
				min(minDiff[2] * inv_dir[2], maxDiff[2] * inv_dir[2])
			), min(minDiff[0] * inv_dir[0], maxDiff[0] * inv_dir[0]));

			//float bt = max(-1.0f, min(minDiff[0] * inv_dir[0], maxDiff[0] * inv_dir[1]));
			//bt = max(bt, min(minDiff[1] * inv_dir[1], maxDiff[1] * inv_dir[1]));
			//bt = max(bt, min(minDiff[2] * inv_dir[2], maxDiff[2] * inv_dir[2]));

			if ((bt > 1e-5 && bt < t_min)
#ifndef USING_FAST_BVH
				|| (minDiff[0] * maxDiff[0] <= 0 &&
				minDiff[1] * maxDiff[1] <= 0 &&
				minDiff[2] * maxDiff[2] <= 0)
#endif
			){
				// reach leaf node of bvh
				if (taid >= 0) {

					TriangleArray& triIndices = tri_arr[taid];
					// #pragma unroll
					for (int j = 1; j < triIndices.triIds[0] + 1; j++) {

						t = triangleIntersectionTest(tris[triIndices.triIds[j]], pathSegment.ray, tmp_intersect, tmp_normal, outside);
						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_index = triIndices.triIds[j];
							hit_geom = false;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
						}
					}
				}
				// keep searching
				else if (sign + 2 < BVH_GPU_STACK_SIZE) {
					arr[sign + 1] = bbox.leftId;
					arr[sign + 2] = bbox.rightId;
					sign += 2;
				}
				//else {
				//	break;
				//}
			}
		}



		if (hit_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_geom ? geoms[hit_index].materialid : tris[hit_index].materialid;
			intersections[path_index].surfaceNormal = normal;
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
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, int depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(depth, idx, iter);
			// thrust::uniform_real_distribution<float> u01(0, 1); // u01(rng) to get random (0, 1)

			Material material = materials[intersection.materialId];

			glm::vec3 intersect = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
			scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = -1;
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
		image[iterationPath.pixelIndex] += iterationPath.color * (float)abs(iterationPath.remainingBounces);
		/*if (iterationPath.remainingBounces == 0) {
			image[iterationPath.pixelIndex] += glm::vec3(1.0f, 0.0f, 1.0f);
		}
		else {
			image[iterationPath.pixelIndex] += iterationPath.color * (float)abs(iterationPath.remainingBounces);
		}*/
		
	}
}

__global__ void markPathSegment(int nPaths, int* bools, int* nbools, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		bool b = iterationPaths[index].remainingBounces > 0;
		bools[index] = b;
		nbools[index] = !b;
	}
}

// for path termination
__global__ void pathSegmentScatter(int n, int scanSum, PathSegment* odata, const PathSegment* idata, const int* bools, const int* indicesPos, const int* indicesNeg) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (bools[index] > 0) {
			odata[indicesPos[index]] = idata[index];
		}
		else {
			odata[indicesNeg[index] + scanSum] = idata[index];
		}
	}
}

// for material sort
__global__ void kernMapMatBitToBoolean(int n, int i, int* bools, int* ebools, const ShadeableIntersection* idata) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		bools[index] = ((idata[index].materialId >> i) & 1);
		ebools[index] = !((idata[index].materialId >> i) & 1);
	}
}

// for path termination
__global__ void pathSegmentAndIntersectionScatter(
	int n, int negCount,
	PathSegment* opaths, const PathSegment* ipaths,
	ShadeableIntersection* ointers, const ShadeableIntersection* iinters,
	const int* bools, const int* indicesPos, const int* indicesNeg) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < n) {
		if (bools[index] > 0) {
			opaths[indicesPos[index] + negCount] = ipaths[index];
			ointers[indicesPos[index] + negCount] = iinters[index];
		}
		else {
			opaths[indicesNeg[index]] = ipaths[index];
			ointers[indicesNeg[index]] = iinters[index];
		}
	}
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
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
	const int blockSize1d = BLOCK_SIZE_1D;
	int depth = 0;
	int num_paths = pixelcount;


#ifdef CACHE_FIRST_BOUNCE
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_pathsBuffer, dev_paths, sizeof(PathSegment) * pixelcount, cudaMemcpyDeviceToDevice);
		checkCUDAError("save dev_pathsBuffer");
	}
	else {
		cudaMemcpy(dev_paths, dev_pathsBuffer, sizeof(PathSegment) * pixelcount, cudaMemcpyDeviceToDevice);
		checkCUDAError("load dev_pathsBuffer");
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif


	//cout << "cam pos = " << cam.position.x << ", " << cam.position.y << ", " << cam.position.z << endl;
	//Ray testRay = Ray();
	//testRay.origin = cam.position;
	//testRay.direction = glm::normalize(glm::vec3(-1, 4, -5) - cam.position);
	//int testTriID = -1;
	//glm::vec3 testIntersectionPoint;
	//glm::vec3 testNormal;
	//bool testOutside;
	//bvhSearch(testRay, testTriID, hst_scene->tris.data(), hst_scene->tris.size(), hst_scene->bvh.data(), hst_scene->bvh.size(),
	//	hst_scene->triArr.data(), hst_scene->triArr.size(), testIntersectionPoint, testNormal, testOutside);
	//cout << "Test: triId = " << testTriID << ", matId = " << hst_scene->tris[testTriID].materialid << endl;


	//Ray testRay = Ray();
	//testRay.origin = cam.position;
	//testRay.direction = glm::normalize(glm::vec3(0.14668556f, 0.66375220f, -0.73342776f));
	//glm::vec3 testIntersectionPoint;
	//glm::vec3 testNormal;
	//bool testOutside;
	//float test_tri_t = triangleIntersectionTest(hst_scene->tris[2], testRay, testIntersectionPoint, testNormal, testOutside);
	//glm::vec3 minDiff = hst_scene->bvh[61].min - testRay.origin;
	//glm::vec3 maxDiff = hst_scene->bvh[61].max - testRay.origin;
	//float test_bt_x = (abs(testRay.direction[0]) < 1e-5) ? -1.0f : glm::max(-1.0f, min(minDiff.x / testRay.direction[0], maxDiff.x / testRay.direction[0]));
	//float test_bt_y = (abs(testRay.direction[1]) < 1e-5) ? -1.0f : glm::max(-1.0f, min(minDiff.y / testRay.direction[1], maxDiff.y / testRay.direction[1]));
	//float test_bt_z = (abs(testRay.direction[2]) < 1e-5) ? -1.0f : glm::max(-1.0f, min(minDiff.z / testRay.direction[2], maxDiff.z / testRay.direction[2]));


	
	// PathSegment* dev_path_end = dev_paths + pixelcount;
	// int num_paths = dev_path_end - dev_paths;
	
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	int mat_num = hst_scene->materials.size();

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		checkCUDAError("cudaMemset dev_intersections");

#ifdef CACHE_FIRST_BOUNCE
		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersectionsBuffer, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
			checkCUDAError("load dev_intersectionsBuffer");
		}
		else {
#endif
			// tracing
#ifdef USING_BVH
			if (hst_scene->bvh.size() > 0) {
				computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths,
					dev_geoms, hst_scene->geoms.size(),
					dev_tris, hst_scene->tris.size(),
					dev_bvh, hst_scene->bvh.size(),
					dev_triArr, hst_scene->triArr.size(),
					dev_intersections);
				checkCUDAError("tcomputeIntersectionsBVH");
			} else {
#endif
				computeIntersectionsNaive << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth, num_paths, dev_paths,
					dev_geoms, hst_scene->geoms.size(),
					dev_tris, hst_scene->tris.size(),
					dev_intersections);
#ifdef USING_BVH
			}
#endif


			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
#ifdef CACHE_FIRST_BOUNCE
			if (iter == 1 && depth == 0) {
				cudaMemcpy(dev_intersectionsBuffer, dev_intersections, sizeof(ShadeableIntersection) * pixelcount, cudaMemcpyDeviceToDevice);
				checkCUDAError("save dev_intersectionsBuffer");
			}
		}
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


		// Sort All Materials
#ifdef MATERIAL_SORT
		if (mat_num > 1) {
			int log2Ceil = 1;
			int product = 1;
			while (product < mat_num) {
				product *= 2;
				log2Ceil++;
			}
			for (int i = 0; i < log2Ceil; i++) {
				kernMapMatBitToBoolean << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, i, dev_bools, dev_nbools, dev_intersections);
				thrust::device_ptr<int> dv_sort_in(dev_bools);
				thrust::device_ptr<int> dv_sort_out(dev_scanBools);
				thrust::exclusive_scan(dv_sort_in, dv_sort_in + num_paths, dv_sort_out);
				thrust::device_ptr<int> dv_sort_nin(dev_nbools);
				thrust::device_ptr<int> dv_sort_nout(dev_scanNBools);
				thrust::exclusive_scan(dv_sort_nin, dv_sort_nin + num_paths, dv_sort_nout);

				int neg_count = -1;
				cudaMemcpy(&neg_count, dev_scanNBools + num_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
				int last_bool = -1;
				cudaMemcpy(&last_bool, dev_bools + num_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
				neg_count += (last_bool == 0);

				pathSegmentAndIntersectionScatter << < numblocksPathSegmentTracing, blockSize1d >> > (
					num_paths, neg_count,
					dev_tempPaths, dev_paths,
					dev_tempIntersections, dev_intersections,
					dev_bools, dev_scanBools, dev_scanNBools);

				PathSegment* tempPath = dev_tempPaths;
				dev_tempPaths = dev_paths;
				dev_paths = tempPath;
				ShadeableIntersection* tempInter = dev_tempIntersections;
				dev_tempIntersections = dev_intersections;
				dev_intersections = tempInter;
			}

		}
#endif



		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, num_paths, depth,
			dev_intersections, dev_paths, dev_materials
		);
		
		
		markPathSegment << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_bools, dev_nbools, dev_paths);
		int lastBool;
		int scanSum;
		cudaMemcpy(&lastBool, dev_bools + pixelcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
		
		thrust::device_ptr<int> dv_in(dev_bools);
		thrust::device_ptr<int> dv_out(dev_scanBools);
		thrust::exclusive_scan(dv_in, dv_in + pixelcount, dv_out);
		cudaMemcpy(&scanSum, dev_scanBools + pixelcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
		scanSum += lastBool;
		

		if (scanSum > 0) {
			thrust::device_ptr<int> dv_nin(dev_nbools);
			thrust::device_ptr<int> dv_nout(dev_scanNBools);
			thrust::exclusive_scan(dv_nin, dv_nin + pixelcount, dv_nout);
			pathSegmentScatter << <numBlocksPixels, blockSize1d >> > (pixelcount, scanSum, dev_tempPaths, dev_paths, dev_bools, dev_scanBools, dev_scanNBools);
			PathSegment* temp = dev_tempPaths;
			dev_tempPaths = dev_paths;
			dev_paths = temp;
		}

#ifdef DEBUG_OUTPUT
		std::cout << "iter-" << iter << ", depth-" << depth << ", paths: " << num_paths << " -> " << scanSum << std::endl;
#endif

		num_paths = scanSum;
		iterationComplete = num_paths <= 0 || depth >= traceDepth;
	
		if (guiData != NULL){ guiData->TracedDepth = depth; }
	}

	checkCUDAError("pathtrace before finalGather");

	// Assemble this iteration and apply it to the image
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	checkCUDAError("pathtrace finalGather");

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	checkCUDAError("pathtrace sendImageToPBO");

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace final cudaMemcpy");
}
