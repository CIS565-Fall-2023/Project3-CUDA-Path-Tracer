#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MATERIAL_SORT 0
#define CACHE_FIRST_BOUNCE 1

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
static ShadeableIntersection* dev_first_bounce_cache = NULL;
static bool first_bounce_cached = false;
static Triangle* dev_tris = NULL;
static BvhNode* dev_bvh_nodes = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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
    checkCUDAError("dev_image");

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    checkCUDAError("dev_paths");

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    checkCUDAError("dev_geoms");

    cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    checkCUDAError("dev_tris");

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    checkCUDAError("dev_materials");

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    checkCUDAError("dev_intersections");

    cudaMalloc(&dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce_cache, 0, pixelcount * sizeof(ShadeableIntersection));
    checkCUDAError("dev_first_bounce_cache");

    cudaMalloc(&dev_bvh_nodes, scene->bvh_nodes.size() * sizeof(BvhNode));
    cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes.data(), scene->bvh_nodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
    
    first_bounce_cached = false;
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_tris);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_first_bounce_cache);
    cudaFree(dev_bvh_nodes);

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

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.no_intersection = false;
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
    , Triangle* tris
    , int tris_size
    , BvhNode* bvh_nodes
    , bool using_bvh
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


        // With BVH, parses through each mesh and tests for intersection through its BVH data structure
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
                if (using_bvh)
				{
                    // BVH Intersection test
					t = intersect_bvh(geom, tris, pathSegment.ray, tmp_intersect, tmp_normal, bvh_nodes, geom.root_node_index);
				}
                else
                {
                    // Naive mesh intersection test
                    t = meshIntersectionTest(geom, pathSegment.ray, tris, tmp_intersect, tmp_normal, outside, tris_size);
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
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].intersectionPoint = intersect_point;
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
    , int depth
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment &curr_segment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            // do a vec3 emittance
            if (material.emittance_vec3 != glm::vec3(-1.0f))
            {
                curr_segment.color *= (materialColor * material.emittance_vec3 * material.color);
                curr_segment.remainingBounces = 0;
            }
            else if (material.emittance > 0.0f) {
                curr_segment.color *= (materialColor * material.emittance * material.color);
                curr_segment.remainingBounces = 0;
            }
            // Material has no emittance
            else
            {
                curr_segment.remainingBounces--;
                scatterRay(curr_segment, intersection.intersectionPoint,
                    intersection.surfaceNormal, material, rng);
            }
        }
        // If there was no intersection, color the ray black.
        else {
            curr_segment.color = glm::vec3(0.0f);
            curr_segment.remainingBounces = 0;
            curr_segment.no_intersection = true;
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

__global__ void cache_intersections(int n, ShadeableIntersection* to, ShadeableIntersection* from)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }
    to[index] = from[index];
}
 
struct ray_not_out_of_bounds
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
        return (!path.no_intersection);
    }

};

struct ray_no_remaining_bounces
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
        return (path.remainingBounces > 0);
    }
};


struct material_order
{
	__host__ __device__
    bool operator()(const ShadeableIntersection& intersection1, const ShadeableIntersection& intersection2)
    {
		return (intersection1.materialId < intersection2.materialId);
	}
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
///////////////////////////////////////////////////////////////////////////
void pathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const int block_size = 64;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
    const dim3 blocks_per_grid_1d((pixelcount + block_size - 1) / block_size);
    dim3 numblocksPathSegmentTracing;
    

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    // thrust set up
    thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
    thrust::device_ptr<PathSegment> new_dev_thrust_path_end(dev_path_end);
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // tracing
#if CACHE_FIRST_BOUNCE
        if (!(depth == 0 && first_bounce_cached))
        {
#endif
            checkCUDAError("before compute_intersections");
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_tris
                , hst_scene->tris.size()
                , dev_bvh_nodes
                , hst_scene->using_bvh()
                );
            checkCUDAError("compute_intersections");
#if CACHE_FIRST_BOUNCE
        }
        else
        {
            cudaMemcpy(dev_intersections, dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
#endif
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#if CACHE_FIRST_BOUNCE
        if (depth == 0 && !first_bounce_cached)
        {
            cudaMemcpy(dev_first_bounce_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            first_bounce_cached = true;
        }
#endif

        depth++;
        // compaction stage one
        new_dev_thrust_path_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, ray_not_out_of_bounds());
        num_paths = new_dev_thrust_path_end - dev_thrust_paths;

#if MATERIAL_SORT
        // make materials contiguous in memory
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, material_order());
#endif
        shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth
            );

        // compaction stage two
        new_dev_thrust_path_end = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, ray_no_remaining_bounces());
        num_paths = new_dev_thrust_path_end - dev_thrust_paths;
        if (num_paths == 0)
        {
            iterationComplete = true; 
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
