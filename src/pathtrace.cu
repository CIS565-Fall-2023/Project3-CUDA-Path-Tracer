#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
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

#define ERRORCHECK 1
#define ENABLE_SORT_BY_MATERIAL 0
#define ENABLE_CACHE_1ST_INTERSECTIONS 1
#define ENABLE_BVH 1
#define ENABLE_DIRECT_LIGHT 1


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

struct pathAlive
{
    __host__ __device__ bool operator()(const PathSegment& pathSegment){
        return pathSegment.remainingBounces > 0;
    }
};

struct compareMatID
{
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b){
        return a.materialId < b.materialId;
    }
};

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
static ShadeableIntersection* dev_1st_intersections_cache = NULL;
static PathSegment* dev_1st_paths_cache = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_BVHNodes = NULL;
static int* dev_BVHTriIdx = NULL;
static glm::vec3* dev_directLights = NULL;
static int* dev_lightIdx = NULL;

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
    cudaMalloc(&dev_1st_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_1st_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_1st_paths_cache, pixelcount * sizeof(PathSegment));
	cudaMemset(dev_1st_paths_cache, 0, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_directLights, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_directLights, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_lightIdx, scene->lightIdx.size() * sizeof(Geom));
	cudaMemcpy(dev_lightIdx, scene->lightIdx.data(), scene->lightIdx.size() * sizeof(int), cudaMemcpyHostToDevice);

	if(scene -> hasMesh) {
	    int triCnt = scene->tri.size();
	    cudaMalloc(&dev_triangles, triCnt * sizeof(Triangle));
	    cudaMemcpy(dev_triangles, scene->tri.data(), triCnt * sizeof(Triangle), cudaMemcpyHostToDevice);
#if ENABLE_BVH
		int bvhCnt = scene->bvhNode.size();
		cudaMalloc(&dev_BVHNodes, bvhCnt * sizeof(BVHNode));
	    cudaMemcpy(dev_BVHNodes, scene->bvhNode.data(), bvhCnt * sizeof(BVHNode), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_BVHTriIdx, triCnt * sizeof(int));
	    cudaMemcpy(dev_BVHTriIdx, scene->triIdx.data(), triCnt * sizeof(int), cudaMemcpyHostToDevice);
#endif
	}
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_1st_intersections_cache);
	cudaFree(dev_1st_paths_cache);
	cudaFree(dev_triangles);
	cudaFree(dev_directLights);
#ifdef ENABLE_BVH
	cudaFree(dev_BVHNodes);
	cudaFree(dev_BVHTriIdx);
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
		segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.L = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.specularBounce = false;

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
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Triangle* triangles
	, BVHNode* bvhNodes
	, int* bvhTriIdx
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
		int meshCnt = 0;

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
#if ENABLE_BVH
                meshCnt++;
#else
			    t = objIntersectionTest(geom, pathSegment.ray, triangles, tmp_intersect, tmp_normal, outside);
#endif
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

#if ENABLE_BVH
        if(meshCnt > 0) {
            int geomIdx = -1;
            t = BVHIntersect(pathSegment.ray, triangles, bvhNodes, bvhTriIdx,
		                 tmp_intersect, tmp_normal, outside, geomIdx);
	        if (t > 0.0f && t_min > t) {
			    t_min = t;
			    hit_geom_index = geomIdx;
			    intersect_point = tmp_intersect;
			    normal = tmp_normal;
		    }
		}
#endif

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

__global__ void computeDirectLight(
	int iter
	, int num_paths
	, PathSegment* pathSegments
	, glm::vec3* directLights
	, Geom* geoms
	, int geoms_size
	, Triangle* triangles
	, BVHNode* bvhNodes
	, int* bvhTriIdx
	, ShadeableIntersection* intersections
	, Material* materials
	, int* lightIdxList
	, int light_size
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		ShadeableIntersection intersection = intersections[path_index];	

		Material mIsect = materials[intersection.materialId];
		if(mIsect.type == SPEC_GLASS || mIsect.type == SPEC_REFL ||
           mIsect.type == SPEC_TRANS) {
		    directLights[path_index] = glm::vec3(0.0f);
		    return;
		}

		glm::vec3 intersect = getPointOnRay(pathSegment.ray, intersection.t);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		int rand = glm::min((int)(u01(rng) * light_size), light_size - 1);
		int lightIdx = lightIdxList[rand];
		Geom light = geoms[lightIdx];

		// Generate ray to a light source

        glm::vec2 xi = glm::vec2(u01(rng), u01(rng)) - glm::vec2(0.5);
        glm::vec3 pos = multiplyMV(light.transform, glm::vec4(xi.x, 0., xi.y, 1.));
        glm::vec3 nor = glm::normalize(multiplyMV(light.invTranspose, glm::vec4(0., 1., 0., 0.)));
        float area = light.scale.x * light.scale.z;

        glm::vec3 wi = pos - intersect;
        glm::vec3 wiW = glm::normalize(wi);

		float dot = glm::dot(wiW, nor);
        float absDot = glm::abs(dot);
        float pdf = (length(wi) * length(wi)) / (absDot * area);

		if (pdf == 0){
		    directLights[path_index] = glm::vec3(0.0f);
			return;
		}

		// Compute if blocked by other objects
		Ray r;
		r.origin = intersect;
		r.direction = wiW;

		float t;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        
		// naive parse through global geoms
		int meshCnt = 0;

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH)
			{
#if ENABLE_BVH
                meshCnt++;
#else
			    t = objIntersectionTest(geom, r, triangles, tmp_intersect, tmp_normal, outside);
#endif
			}

			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
			}
		}

#if ENABLE_BVH
        if(meshCnt > 0) {
            int geomIdx = -1;
            t = BVHIntersect(r, triangles, bvhNodes, bvhTriIdx,
		                 tmp_intersect, tmp_normal, outside, geomIdx);
	        if (t > 0.0f && t_min > t)
		    {
			    t_min = t;
			    hit_geom_index = geomIdx;
		    }
		}
#endif			

		if (hit_geom_index == lightIdx) {		    
			Material m = materials[light.materialid];
            float absDot = glm::abs(glm::dot(intersection.surfaceNormal, wiW));
			float INV_PI = 0.31830988618379067;
			glm::vec3 f = mIsect.color * INV_PI;
			glm::vec3 Li = (float)light_size * m.color * m.emittance;
			directLights[path_index] = f * Li * absDot / pdf;             
		}
		else {
		    directLights[path_index] = glm::vec3(0.0f);
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
	int iter, int bounces
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
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		if (intersection.t > 0.0f) { 

			Material material = materials[intersection.materialId];
			if (material.emittance > 0.0f) {
				pathSegments[idx].L += pathSegments[idx].throughput * (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
			
				glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
				glm::vec3 normal = intersection.surfaceNormal;
				scatterRay(pathSegments[idx], intersect, normal, material, rng);
				pathSegments[idx].remainingBounces --;
			}
		}
		else {
		    pathSegments[idx].remainingBounces = 0;
		}

		// Russian roulette
		if (bounces > 3 && pathSegments[idx].remainingBounces > 0) {	
		    float y = glm::max(pathSegments[idx].throughput.r, pathSegments[idx].throughput.g);
			y = glm::max(y, pathSegments[idx].throughput.b);
		    float q = glm::max(0.05f, 1 - y);
		    if (u01(rng) < q) {
			    pathSegments[idx].remainingBounces = 0;
		    }
		    else {
			    pathSegments[idx].throughput /= (1 - q);
		    }
		}
	}
}

__global__ void shadeMaterialDirectLighting(
	int iter
	, int bounces
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* geoms
	, glm::vec3* directLights
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{	    
		ShadeableIntersection intersection = shadeableIntersections[idx];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		if (intersection.t > 0.0f) { 
			Material material = materials[intersection.materialId];
		
			if (material.emittance > 0.0f) {
				if(bounces == 0 || pathSegments[idx].specularBounce){
				    pathSegments[idx].L += pathSegments[idx].throughput * (material.color * material.emittance);
			    }
				pathSegments[idx].remainingBounces = 0;
				return;
			}			

			glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
			glm::vec3 normal = intersection.surfaceNormal;

			if(material.type == SPEC_GLASS || material.type == SPEC_REFL ||
               material.type == SPEC_TRANS || material.type == PLASTIC){
			    pathSegments[idx].specularBounce = true;
			}
			else{
			    pathSegments[idx].specularBounce = false;			
			}

			if(material.type == DIFFUSE || material.type == PLASTIC){
			    pathSegments[idx].L += pathSegments[idx].throughput * directLights[idx];
			}
		
			scatterRay(pathSegments[idx], intersect, normal, material, rng);
			pathSegments[idx].remainingBounces --;
			
		}
		else {
		    pathSegments[idx].remainingBounces = 0;
			// Env color
			//pathSegments[idx].L += pathSegments[idx].throughput * glm::vec3(0.25, 0.25, 0.25);
		}

		// Russian roulette
		if (bounces > 3 && pathSegments[idx].remainingBounces > 0) {	
		    float y = glm::max(pathSegments[idx].throughput.r, pathSegments[idx].throughput.g);
			y = glm::max(y, pathSegments[idx].throughput.b);
		    float q = glm::max(0.05f, 1 - y);
		    if (u01(rng) < q) {
			    pathSegments[idx].remainingBounces = 0;
		    }
		    else {
			    pathSegments[idx].throughput /= (1 - q);
		    }
		}
	}
}

// This only draws the direct lighting in the scene
__global__ void shadeMaterialDirectLightingTest(
	int iter
	, int bounces
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* geoms
	, glm::vec3* directLights
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{	    
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { 

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];

			
			if (material.emittance > 0.0f) {
				pathSegments[idx].L += pathSegments[idx].throughput * (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0;
				return;
			}
			
			pathSegments[idx].L += directLights[idx];
			pathSegments[idx].remainingBounces = 0;
			
		}
		else {
		    pathSegments[idx].remainingBounces = 0;
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
		image[iterationPath.pixelIndex] += iterationPath.L;
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

#if ENABLE_CACHE_1ST_INTERSECTIONS
    if(iter == 1){
	    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	    checkCUDAError("generate camera ray");
		cudaMemcpy(dev_1st_paths_cache, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else{
	    cudaMemcpy(dev_paths, dev_1st_paths_cache, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
#else
     generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	 checkCUDAError("generate camera ray");	
#endif
	
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	bool iterationComplete = false;

	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if ENABLE_CACHE_1ST_INTERSECTIONS
        if (depth == 0) {
            if(iter == 1){
			    computeIntersections << <numblocksPathSegmentTracing, blockSize1d>> > (
                depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_triangles, dev_BVHNodes, dev_BVHTriIdx
				, dev_intersections
				);
                cudaDeviceSynchronize();
                cudaMemcpy(dev_1st_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
			    cudaMemcpy(dev_intersections, dev_1st_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}          
        } 
		else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_triangles, dev_BVHNodes, dev_BVHTriIdx
			, dev_intersections
			);
		    cudaDeviceSynchronize();
        }
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_triangles, dev_BVHNodes, dev_BVHTriIdx
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif
		

#if SORT_MATERIALS
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMatID());
#endif

#if ENABLE_DIRECT_LIGHT
        computeDirectLight << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter
			, num_paths
			, dev_paths
			, dev_directLights
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_triangles, dev_BVHNodes, dev_BVHTriIdx
			, dev_intersections
			, dev_materials
			, dev_lightIdx
			, hst_scene->lightIdx.size()
			);

        shadeMaterialDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms,
			dev_directLights
			);

#else
		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
#endif	
        depth++;
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, pathAlive());
        num_paths = dev_path_end - dev_paths;

		iterationComplete = num_paths <= 0 || depth >= traceDepth;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
