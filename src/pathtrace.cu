#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "material.h"

#define ERRORCHECK 1


template<typename T>
void checkCudaMem(T* d_ptr, int size) {
    T* h_ptr = new T[size];
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    delete[] h_ptr;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image, bool acesFilm, bool NoGammaCorrection) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;
        if (acesFilm)
            pix = pix * (pix * (pix * 2.51f + 0.03f) + 0.024f) / (pix * (pix * 3.7f + 0.078f) + 0.14f);

        if (!NoGammaCorrection)
            pix = glm::pow(pix, glm::vec3(1.f / 2.2f));

        glm::ivec3 color = glm::ivec3(glm::clamp(pix, 0.f, 1.f) * 255.0f);
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
static TriangleDetail* dev_geoms = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static PathSegment* dev_paths_terminated = nullptr;
static int* dev_materialIsectIndices = nullptr;
static int* dev_materialIsectIndicesCache = nullptr;
static int* dev_materialSegIndices = nullptr;
static TBVHNode* dev_tbvhNodes = nullptr;
static thrust::device_ptr<PathSegment> dev_paths_thrust;
static thrust::device_ptr<PathSegment> dev_paths_terminated_thrust;
static thrust::device_ptr<int> dev_materialIsectIndices_thrust;
static thrust::device_ptr<int> dev_materialSegIndices_thrust;
static ShadeableIntersection* dev_intersections = nullptr;
static ShadeableIntersection* dev_intersections_cache = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData, Scene* scene)
{
    guiData = imGuiData;
    guiData->TracedDepth = scene->state.traceDepth;
    guiData->SortByMaterial = false;
    guiData->UseBVH = true;
    guiData->ACESFilm = false;
    guiData->NoGammaCorrection = false;
    guiData->focalLength = scene->state.camera.focalLength;
    guiData->apertureSize = scene->state.camera.apertureSize;
    guiData->theta = 0.f;
    guiData->phi = 0.f;
    guiData->cameraLookAt = scene->state.camera.lookAt;
    guiData->zoom = 1.f;
}

void UpdateDataContainer(GuiDataContainer* imGuiData, Scene* scene, float zoom, float theta, float phi)
{
    imGuiData->TracedDepth = scene->state.traceDepth;
    imGuiData->SortByMaterial = false;
    imGuiData->UseBVH = true;
    imGuiData->ACESFilm = false;
    imGuiData->NoGammaCorrection = false;
    imGuiData->focalLength = scene->state.camera.focalLength;
    imGuiData->apertureSize = scene->state.camera.apertureSize;
    imGuiData->theta = theta;
    imGuiData->phi = phi;
    imGuiData->cameraLookAt = scene->state.camera.lookAt;
    imGuiData->zoom = zoom;
}

void pathtraceInit(Scene* scene) {
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_terminated, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(TriangleDetail));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(TriangleDetail), cudaMemcpyHostToDevice);

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

    cudaMalloc(&dev_tbvhNodes, 6 * hst_scene->tbvh.nodesNum * sizeof(TBVHNode));
    for (int i = 0; i < 6; i++)
    {
        cudaMemcpy(dev_tbvhNodes + i * hst_scene->tbvh.nodesNum, hst_scene->tbvh.nodes[i].data(), hst_scene->tbvh.nodesNum * sizeof(TBVHNode), cudaMemcpyHostToDevice);
    }

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, Scene::Settings::CameraSettings settings)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        float rx = 0.f;
        float ry = 0.f;
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments->remainingBounces);
        thrust::uniform_real_distribution<float> u(-0.5, 0.5);
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.f);
        segment.throughput = glm::vec3(1.f);
        if (settings.antiAliasing) {
            rx = u(rng);
            ry = u(rng);
        }

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + rx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + ry - (float)cam.resolution.y * 0.5f));
        if (settings.dof) {
            glm::vec3 forward = glm::cross(cam.up, cam.right);
            float t = cam.focalLength / AbsDot(segment.ray.direction, forward);
            glm::vec3 randPt = cam.apertureSize * squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
            glm::vec3 tmpRayOrigin = cam.position + randPt.x * cam.right + randPt.y * cam.up;
            glm::vec3 focusPoint = segment.ray.origin + t * segment.ray.direction;
            segment.ray.direction = glm::normalize(focusPoint - tmpRayOrigin);
            segment.ray.origin = tmpRayOrigin;
        }

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
    , TriangleDetail* geoms
    , TBVHNode* nodes
    , int geoms_size
    , int nodesNum
    , ShadeableIntersection* intersections
    , bool sortByMaterial
    , int* materialIndices
    , bool useBVH
    , cudaTextureObject_t cubemap
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float3 tmp_t;
        float3 t;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        // naive parse through global geoms
        if (useBVH)
            tmp_t = sceneIntersectionTest(geoms, nodes, nodesNum, pathSegment.ray, hit_geom_index, cubemap);
        else {
            for (int i = 0; i < geoms_size; i++)
            {
                TriangleDetail& tri = geoms[i];
                t = triangleIntersectionTest(tri, pathSegment.ray);
                // TODO: add more intersection tests here... triangle? metaball? CSG?

                // Compute the minimum t from the intersection tests to determine what
                // scene geometry object was hit first.
                if (t.x > 0.0f && t_min > t.x)
                {
                    tmp_t = t;
                    t_min = t.x;
                    hit_geom_index = i;
                }
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
            t = tmp_t;
            t_min = t.x;
            float w = 1 - t.y - t.z;
            TriangleDetail& tri = geoms[hit_geom_index];
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = tri.materialid;
            intersections[path_index].uv = t.y * tri.uv0 + t.z * tri.uv1 + w * tri.uv2;
            intersections[path_index].pos = getPointOnRay(pathSegment.ray, t_min);
            intersections[path_index].woW = -pathSegment.ray.direction;
            intersections[path_index].surfaceNormal =
                glm::normalize(multiplyMV(tri.t.invTranspose, glm::vec4(t.y * tri.normal0 + t.z * tri.normal1 + w * tri.normal2, 0.f)));
            glm::vec4 tangent = t.y * tri.tangent0 + t.z * tri.tangent1 + w * tri.tangent2;
            float tmpTanw = tangent[3];
            tangent[3] = 0.f;
            intersections[path_index].tangent = glm::vec4(glm::normalize(multiplyMV(tri.t.invTranspose, tangent)), tmpTanw);
            if (sortByMaterial)
                materialIndices[path_index] = intersections[path_index].materialId;
        }
    }
}

__global__ void shadeMaterial(
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , cudaTextureObject_t envMap
    , Scene::Settings::TransferableSettings settings
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pSeg.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray
            glm::vec3 emissiveColor{ 0.f };
            if (material.emissiveTexture.index > 0) {
                emissiveColor = sampleTexture(material.emissiveTexture.cudaTexObj, intersection.uv);
            }
            if (material.type == Material::Type::LIGHT) {
                pSeg.color = pSeg.throughput * (glm::vec3(material.pbrMetallicRoughness.baseColorFactor) * material.emissiveFactor * material.emissiveStrength);
                pSeg.remainingBounces = 0;
            }
            else if (glm::length(emissiveColor) > EPSILON) {
                pSeg.color = pSeg.throughput * emissiveColor * material.emissiveFactor * material.emissiveStrength;
                pSeg.remainingBounces = 0;
            }
            else {
                if (settings.testNormal) {
                    glm::vec3 nor = intersection.surfaceNormal;
                    if (material.normalTexture.index != -1) {
                        nor = sampleTexture(material.normalTexture.cudaTexObj, intersection.uv);
                        nor = glm::normalize(nor) * 2.f - 1.f;
                        nor = glm::normalize((glm::mat3(glm::vec3(intersection.tangent), glm::cross(intersection.surfaceNormal, glm::vec3(intersection.tangent)) * intersection.tangent[3], intersection.surfaceNormal)) * nor);
                    }
                    pSeg.color = nor * 0.5f + 0.5f;
                    pSeg.remainingBounces = 0;
                }
                else if (settings.testIntersect) {
                    pSeg.color = settings.testColor;
                    pSeg.remainingBounces = 0;
                }
                else {
                    float ao{ 1.f };
                    if (material.occlusionTexture.index != -1) {
                        auto aoData = tex2D<float4>(material.occlusionTexture.cudaTexObj, intersection.uv.x, intersection.uv.y);
                        ao = aoData.x;
                    }
                    glm::vec3 nor = intersection.surfaceNormal;
                    if (material.normalTexture.index != -1) {
                        nor = sampleTexture(material.normalTexture.cudaTexObj, intersection.uv);
                        nor = glm::normalize(nor) * 2.f - 1.f;
                        nor = glm::normalize((glm::mat3(glm::vec3(intersection.tangent), glm::cross(intersection.surfaceNormal, glm::vec3(intersection.tangent)) * intersection.tangent[3], intersection.surfaceNormal)) * nor);
                    }
                    BsdfSample sample;
                    auto bsdf = ao * sample_f(material, settings.isProcedural, settings.scale, nor, intersection.uv, intersection.woW, glm::vec3(u01(rng), u01(rng), u01(rng)), sample);
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
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else if (settings.envMapEnabled) {
            pSeg.color += pSeg.throughput * sampleEnvTexture(envMap, sampleSphericalMap(pSeg.ray.direction));
            pSeg.remainingBounces = 0;
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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, hst_scene->settings.camSettings);
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
        checkCudaMem(dev_tbvhNodes, 6 * hst_scene->tbvh.nodesNum);
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , dev_tbvhNodes
            , hst_scene->geoms.size()
            , hst_scene->tbvh.nodesNum
            , dev_intersections_cache
            , guiData->SortByMaterial
            , dev_materialIsectIndices
            , guiData->UseBVH
            , hst_scene->cubemap.texObj
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
                , dev_tbvhNodes
                , hst_scene->geoms.size()
                , hst_scene->tbvh.nodesNum
                , dev_intersections
                , guiData->SortByMaterial
                , dev_materialIsectIndices
                , guiData->UseBVH
                , hst_scene->cubemap.texObj
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            if (guiData->SortByMaterial) {
                cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
                thrust::sort_by_key(dev_materialIsectIndices_thrust, dev_materialIsectIndices_thrust + num_paths, dev_intersections);
                thrust::sort_by_key(dev_materialSegIndices_thrust, dev_materialSegIndices_thrust + num_paths, dev_paths_thrust);
            }
        }

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
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            hst_scene->envMapTexture.cudaTexObj,
            hst_scene->settings.trSettings);
        depth++;
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
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, guiData->ACESFilm, guiData->NoGammaCorrection);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
