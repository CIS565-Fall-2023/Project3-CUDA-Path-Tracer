#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "common.h"
#include "sampler.h"

struct CompactTerminatedPaths {
	CPU_GPU bool operator() (const PathSegment& segment) {
		return !(segment.pixelIndex >= 0 && segment.IsEnd());
	}
};

struct RemoveInvalidPaths {
	CPU_GPU bool operator() (const PathSegment& segment) {
		return segment.pixelIndex < 0 || segment.IsEnd();
	}
};

CPU_ONLY CudaPathTracer::~CudaPathTracer()
{
	// free ptr
	SafeCudaFree(dev_hdr_img);  // no-op if dev_image is null
	SafeCudaFree(dev_paths);
	SafeCudaFree(dev_geoms);
	SafeCudaFree(dev_materials);
	SafeCudaFree(dev_intersections);

	if (cuda_pbo_dest_resource)
	{
		UnRegisterPBO();
	}

	checkCUDAError("CudaPathTracer delete Error!");
}

CPU_ONLY GPU_ONLY thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

CPU_GPU void writePixel(glm::vec3& hdr_pixel, uchar4& pixel)
{
	// tone mapping
	hdr_pixel = hdr_pixel / (1.f + hdr_pixel);

	// gammar correction
	hdr_pixel = glm::pow(hdr_pixel, glm::vec3(1.f / 2.2f));

	// map to [0, 255]
	hdr_pixel = glm::mix(glm::vec3(0.f), glm::vec3(255.f), hdr_pixel);
	
	// write color
	pixel = { static_cast<unsigned char>(hdr_pixel.r), 
			  static_cast<unsigned char>(hdr_pixel.g), 
			  static_cast<unsigned char>(hdr_pixel.b), 
			  255 };
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image) 
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= resolution.x || y >= resolution.y) return;

	int index = x + (y * resolution.x);
	glm::vec3 pix = image[index];

	writePixel(pix, pbo[index]);
}

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
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
		segment.ray = cam.CastRay({x, y});

		segment.throughput = glm::vec3(1.0f);
		segment.radiance = glm::vec3(0.0f);

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
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		if (pathSegment.IsEnd()) 
		{
			intersections[path_index].t = -1.0f;
			return;
		}
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
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].surfacePosition = intersect_point;
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
				pathSegments[idx].throughput *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].throughput *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].throughput *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].throughput = glm::vec3(0.0f);
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(float u, int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nPaths) return;

	PathSegment iterationPath = iterationPaths[index];
	glm::vec3 pre_color = image[iterationPath.pixelIndex];
	glm::vec3 new_color = glm::mix(pre_color, iterationPath.radiance, u);

	image[iterationPath.pixelIndex] = new_color;
}

// Naive BSDF sample only
__global__ void KernelNaiveGI(int iteration, int num_paths, 
							ShadeableIntersection* shadeableIntersections,
							PathSegment* pathSegments,
							Material* materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	PathSegment segment = pathSegments[idx];

	if (intersection.t > 0.f && !segment.IsEnd())
	{
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;
		//pathSegments[idx].radiance = intersection.surfaceNormal * 0.5f + 0.5f;
		if (material.emittance > 0.0f) {
			glm::vec3 final_throughput = pathSegments[idx].throughput * material.emittance;
			pathSegments[idx].radiance = final_throughput;
			pathSegments[idx].Terminate();
		}
		else
		{		
			glm::vec3 wo = WorldToLocal(intersection.surfaceNormal) * -segment.ray.direction;
			if (wo.z < 0.f)
			{
				pathSegments[idx].Terminate();
				return;
			}

			thrust::default_random_engine rng = makeSeededRandomEngine(iteration, idx, 0);
			thrust::uniform_real_distribution<float> u01(0.f, 1.f);

			// naive diffuse surface
			glm::vec3 wi = SquareToHemisphereCosine({ u01(rng), u01(rng) });
			glm::vec3 wiW = glm::normalize(LocalToWorld(intersection.surfaceNormal) * wi);
		
			float pdf = SquareToHemisphereCosinePDF(wi);
		
			// generate new ray
			pathSegments[idx].ray = Ray::SpawnRay(intersection.surfacePosition, wiW);
			pathSegments[idx].throughput *= materialColor * InvPi;// *wi.z / pdf;
		}
	}
	else
	{
		pathSegments[idx].Terminate();
	}
}

CPU_ONLY void CudaPathTracer::Init(Scene* scene)
{
	m_Iteration = 0;
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_hdr_img, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_hdr_img, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_terminated_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need

	thrust_dev_paths = thrust::device_ptr<PathSegment>(dev_paths);
	thrust_dev_terminated_paths = thrust::device_ptr<PathSegment>(dev_terminated_paths);

	checkCUDAError("pathtraceInit");
}

CPU_ONLY void CudaPathTracer::GetImage(uchar4* host_image)
{
	const Camera& cam = hst_scene->state.camera;
	// Retrieve image from GPU
	cudaMemcpy(host_image, dev_img, cam.resolution.x * cam.resolution.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

CPU_ONLY void CudaPathTracer::RegisterPBO(unsigned int pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_dest_resource, pbo, cudaGraphicsMapFlagsNone);
	size_t byte_count = resolution.x * resolution.y * 4 * sizeof(uchar4);
	cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dev_img, &byte_count, cuda_pbo_dest_resource);
	checkCUDAError("Get PBO pointer Error");
}

CPU_ONLY void CudaPathTracer::Render()
{
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// TODO: might change to dynamic block size
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, m_Iteration, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (depth < 5 && num_paths > 0) {
		depth++;

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
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

		//shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
		//	iter,
		//	num_paths,
		//	dev_intersections,
		//	dev_paths,
		//	dev_materials
		//	);
		//iterationComplete = true; // TODO: should be based off stream compaction results.
		KernelNaiveGI << <numblocksPathSegmentTracing, blockSize1d >> > (m_Iteration, num_paths,
			dev_intersections, dev_paths, dev_materials);

		if (guiData != nullptr)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

	float u = 1.f / static_cast<float>(m_Iteration + 1); // used for interpolation between last frame and this frame

	finalGather << <numBlocksPixels, blockSize1d >> > (u, num_paths, dev_hdr_img, dev_paths);
	checkCUDAError("Final Gather failed");
	cudaDeviceSynchronize();
	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (dev_img, cam.resolution, dev_hdr_img);

	checkCUDAError("pathtrace");
	++m_Iteration;
}