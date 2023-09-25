#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "material.h"

#define ANTIALIASING 0
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

template<typename T>
void checkCudaMem(T* d_ptr, int size) {
    T* h_ptr = new T[size];
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    delete[] h_ptr;
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

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;
static Geom* dev_geoms = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static PathSegment* dev_paths_terminated = nullptr;
static int* dev_materialIsectIndices = nullptr;
static int* dev_materialIsectIndicesCache = nullptr;
static int* dev_materialSegIndices = nullptr;
static thrust::device_ptr<PathSegment> dev_paths_thrust;
static thrust::device_ptr<PathSegment> dev_paths_terminated_thrust;
static thrust::device_ptr<int> dev_materialIsectIndices_thrust;
static thrust::device_ptr<int> dev_materialSegIndices_thrust;
static ShadeableIntersection* dev_intersections = nullptr;
static ShadeableIntersection* dev_intersections_cache = nullptr;
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

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_terminated, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialIsectIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_materialSegIndices, pixelcount * sizeof(int));

    cudaMalloc(&dev_materialIsectIndicesCache, pixelcount * sizeof(int));
    cudaMalloc(&dev_materialIsectIndicesCache, pixelcount * sizeof(int));

    dev_materialIsectIndices_thrust = thrust::device_ptr<int>(dev_materialIsectIndices);
    dev_materialSegIndices_thrust = thrust::device_ptr<int>(dev_materialSegIndices);

    dev_paths_thrust = thrust::device_ptr<PathSegment>(dev_paths);
    dev_paths_terminated_thrust = thrust::device_ptr<PathSegment>(dev_paths_terminated);
    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is nullptr
    cudaFree(dev_paths);
    cudaFree(dev_paths_terminated);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialIsectIndices);
    cudaFree(dev_materialIsectIndicesCache);
    cudaFree(dev_materialSegIndices);
    cudaFree(dev_intersections_cache);
    // TODO: clean up any extra device memory you created

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
        float rx = 0.f;
        float ry = 0.f;
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
#if ANTIALIASING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u(-0.5, 0.5);
            rx = u(rng);
            ry = u(rng);
#endif // ANTIALIASING
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.f);
        segment.throughput = glm::vec3(1.f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + rx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + ry - (float)cam.resolution.y * 0.5f)
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
    , ShadeableIntersection* intersections
    , bool sortByMaterial,
    int* materialIndices
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
            materialIndices[path_index] = -1;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].pos = intersect_point;
            intersections[path_index].woW = -pathSegment.ray.direction;
            if (sortByMaterial)
                materialIndices[path_index] = intersections[path_index].materialId;
        }
    }
}

__global__ void shadeMaterial(
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
        PathSegment& pSeg = pathSegments[idx];
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
                pSeg.color = pSeg.throughput * (materialColor * material.emittance);
                pSeg.remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                BsdfSample sample;
                auto bsdf = sample_f(material, intersection.surfaceNormal, intersection.woW, glm::vec3(u01(rng), u01(rng), u01(rng)), sample);
                if (sample.pdf <= 0) {
                    pSeg.remainingBounces = 0;
                    pSeg.pixelIndex = -1;
                }
                else {
                    pSeg.remainingBounces -= 1;
                    pSeg.throughput *= bsdf / sample.pdf * AbsDot(intersection.surfaceNormal, sample.wiW);
                    pSeg.ray = SpawnRay(intersection.pos, sample.wiW);
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pSeg.color = glm::vec3(0.0f);
            pSeg.remainingBounces = 0;
            pSeg.pixelIndex = -1;
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
    auto dev_paths_terminated_end = dev_paths_terminated_thrust;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    if (!hst_scene->state.isCached) {
        hst_scene->state.isCached = true;
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections_cache
            , guiData->SortByMaterial,
            dev_materialIsectIndices
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        if (guiData->SortByMaterial) {
            cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_materialIsectIndicesCache, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
            thrust::sort_by_key(dev_materialIsectIndices_thrust, dev_materialIsectIndices_thrust + num_paths, dev_intersections_cache);
        }
    }

    bool iterationComplete = false;
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    while (!iterationComplete) {

        // tracing
        if (depth == 0 && hst_scene->state.isCached) {
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndicesCache, pixelcount * sizeof(int), cudaMemcpyDeviceToDevice);
            thrust::sort_by_key(dev_materialSegIndices_thrust, dev_materialSegIndices_thrust + pixelcount, dev_paths_thrust);
        }
        else {
            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , guiData->SortByMaterial,
                dev_materialIsectIndices
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            if (guiData->SortByMaterial) {
                cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
                thrust::sort_by_key(dev_materialIsectIndices_thrust, dev_materialIsectIndices_thrust + num_paths, dev_intersections);
                thrust::sort_by_key(dev_materialSegIndices_thrust, dev_materialSegIndices_thrust + num_paths, dev_paths_thrust);
            }
        }
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
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );
        // gather valid terminated paths
        dev_paths_terminated_end = thrust::remove_copy_if(dev_paths_thrust, dev_paths_thrust + num_paths, dev_paths_terminated_end,
            [] __host__  __device__(const PathSegment & p) { return !(p.pixelIndex >= 0 && p.remainingBounces == 0); });
        int num_paths_valid = dev_paths_terminated_end - dev_paths_terminated_thrust;
        auto end = thrust::remove_if(dev_paths_thrust, dev_paths_thrust + num_paths,
            [] __host__  __device__(const PathSegment & p) { return p.pixelIndex < 0 || p.remainingBounces <= 0; });
        num_paths = end - dev_paths_thrust;

        iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.

        if (guiData != nullptr)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    int num_paths_valid = dev_paths_terminated_end - dev_paths_terminated_thrust;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths_valid, dev_image, dev_paths_terminated);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
