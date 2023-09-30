#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "bvh.h"

#define ERRORCHECK 1
#define DEBUG 1
#define GATHER 1
#define BVH 1

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


struct is_black
{
	__host__ __device__
		bool operator()(const PathSegment& seg)
	{
		return seg.color == glm::vec3(0.);
	}
};


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, float iter, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
#if GATHER
		image[iterationPath.pixelIndex] = glm::mix(image[iterationPath.pixelIndex],iterationPath.color,1.f/iter);
#else
		image[iterationPath.pixelIndex] = image[iterationPath.pixelIndex] + iterationPath.color;
#endif
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image, float iter) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = glm::pow(image[index],glm::vec3(1/2.2));

		glm::ivec3 color;
#if GATHER
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
#else
		color.x = glm::clamp((int)(pix.x * 255.0 / iter), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0 / iter), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0 / iter), 0, 255);
#endif
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void transformTriangles(int num_trigs, Mesh* in_meshs, Triangle* out_trigs) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= num_trigs)return;
	Triangle& trig = out_trigs[idx];
	Mesh& mesh = in_meshs[trig.meshId];
	
	trig.v1.pos = glm::vec3(mesh.transform * glm::vec4(trig.v1.pos,1));
	trig.v1.normal = glm::normalize(glm::mat3(mesh.invTranspose) * trig.v1.normal);

	trig.v2.pos = glm::vec3(mesh.transform * glm::vec4(trig.v2.pos, 1));
	trig.v2.normal = glm::normalize(glm::mat3(mesh.invTranspose) * trig.v2.normal);

	trig.v3.pos = glm::vec3(mesh.transform * glm::vec4(trig.v3.pos, 1));
	trig.v3.normal = glm::normalize(glm::mat3(mesh.invTranspose) * trig.v3.normal);
}

void PathTracer::initMeshTransform()
{
	cout << "Transform triangles based on transformation matrix" << endl;
	int blockSize1d = 128;
	int N = dev_trigs.size();
	dim3 numblocksTransformTrigs = (N + blockSize1d - 1) / blockSize1d;
	transformTriangles<<<numblocksTransformTrigs,blockSize1d>>>(N, dev_geoms.get(), dev_trigs.get());
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void PathTracer::initDataContainer(GuiDataContainer* guiData)
{
	m_guiData = guiData;
}
void PathTracer::pathtraceInit(Scene* scene)
{
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	this->dev_img.malloc(pixelcount, "Malloc dev_img error");
	cudaMemset(this->dev_img.get(), 0, pixelcount * sizeof(glm::vec3));
	
	this->dev_path.malloc(pixelcount, "Malloc dev_path error");

	this->dev_geoms.malloc(scene->meshs.size(), "Malloc dev_mesh error");
	cudaMemcpy(this->dev_geoms.get(), scene->meshs.data(), scene->meshs.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

	this->dev_trigs.malloc(scene->trigs.size(), "Malloc dev_trigs error");
	cudaMemcpy(this->dev_trigs.get(), scene->trigs.data(), scene->trigs.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	initMeshTransform();
	//get transformed triangles to CPU to build BVH tree, triangles will be reordered based on tree
	std::vector<Triangle> tmp_trig(scene->trigs.size());
	cudaMemcpy(tmp_trig.data(), this->dev_trigs.get(), scene->trigs.size() * sizeof(Triangle), cudaMemcpyDeviceToHost);
	BVHTreeBuilder builder;
	auto bvh = builder.buildBVHTree(tmp_trig);
	//sent ordered triangle back to GPU
	cudaMemcpy(this->dev_trigs.get(), tmp_trig.data(), scene->trigs.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	this->dev_bvh.malloc(bvh.size(), "Malloc dev_bvh error");
	cudaMemcpy(this->dev_bvh.get(), bvh.data(), bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
	

	this->dev_mat.malloc(scene->materials.size(), "Malloc material error");
	cudaMemcpy(this->dev_mat.get(), scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	this->dev_intersect.malloc(pixelcount, "Malloc dev_intersect error");
	cudaMemset(dev_intersect.get(), 0, pixelcount * sizeof(ShadeableIntersection));

	checkCUDAError("pathtraceInit");
}


// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* in_paths
	, Triangle* in_trigs
	, Mesh* in_meshs
	, int num_trigs
	, ShadeableIntersection* out_intersects
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment seg = in_paths[path_index];

		float t;
		float t_min = FLT_MAX;
		
		glm::vec3 bary; // for interpolation
		glm::vec3 tmp_bary;
		int hit_trig_index = -1;
		//Triangle hit_trig;
		// naive parse through global geoms
		for (int i = 0; i < num_trigs; i++)
		{
			const Triangle& trig = in_trigs[i];

			bool hit = triangleIntersectionTest(seg.ray.origin, seg.ray.direction, trig.v1.pos, trig.v2.pos, trig.v3.pos, tmp_bary, t);
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (hit && t_min > t)
			{
				t_min = t;
				bary = tmp_bary;
				hit_trig_index = i;
			}
		}
		if (hit_trig_index == -1)
		{
			out_intersects[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			const Triangle& trig = in_trigs[hit_trig_index];
			out_intersects[path_index].t = t_min;
			out_intersects[path_index].bary = bary;
			out_intersects[path_index].materialId = in_meshs[trig.meshId].materialId;
			out_intersects[path_index].surfaceNormal = glm::normalize(trig.v1.normal * bary.x + trig.v2.normal * bary.y + trig.v3.normal * bary.z);
		}
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* in_paths
	, Triangle* in_trigs
	, Mesh* in_meshs
	, int num_trigs
	, BVHNode* in_bvh
	, int num_bvh
	, ShadeableIntersection* out_intersects
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment seg = in_paths[path_index];
		
		float t;
		float t_min = 100000.f;
		
		glm::vec3 bary; // for interpolation
		glm::vec3 tmp_bary;
		int hit_trig_index = -1;
		int needToVisit[33];
		int stackTop = 0;
		needToVisit[0] = 0;

		while (stackTop >= 0 && stackTop < 33) {
			int idxToVis = needToVisit[stackTop--];
			BVHNode node = in_bvh[idxToVis];
			bool hit = AABBIntersectionTest(seg.ray.origin, seg.ray.direction, node.boundingBox,t);
			if (hit && t_min > t) 
			{ 
				if (node.primNum == 0) {
					//second child
					int firstChild = idxToVis + 1;
					int secondChild = -1;
					if (node.secondChildOffset != -1) {
						secondChild = idxToVis + node.secondChildOffset;
						if (seg.ray.direction[node.axis] > 0) {
							needToVisit[++stackTop] = secondChild;
							needToVisit[++stackTop] = firstChild;
						}
						else {
							needToVisit[++stackTop] = firstChild;
							needToVisit[++stackTop] = secondChild;
						}
					}
					else {
						needToVisit[++stackTop] = firstChild;
					}
					
				}
				else { //every thing in bounding box will have larger t
					for (int i = 0;i < node.primNum;++i) {
						const Triangle& trig = in_trigs[node.firstPrimId + i];
						hit = triangleIntersectionTest(seg.ray.origin, seg.ray.direction, trig.v1.pos, trig.v2.pos, trig.v3.pos, tmp_bary,t);
						if (hit && t_min > t)
						{
							t_min = t;
							bary = tmp_bary;
							hit_trig_index = node.firstPrimId + i;
						}
					}
				}
			}
		}

		if (hit_trig_index == -1)
		{
			out_intersects[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			const Triangle& trig = in_trigs[hit_trig_index];
			out_intersects[path_index].t = t_min;
			out_intersects[path_index].bary = bary;
			out_intersects[path_index].materialId = in_meshs[trig.meshId].materialId;
			out_intersects[path_index].surfaceNormal = glm::normalize(trig.v1.normal * bary.x + trig.v2.normal * bary.y + trig.v3.normal * bary.z);
		}
	}
}



__global__ void processPBR(
	int iter, int depth
	, int num_paths
	, ShadeableIntersection* intersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_idx >= num_paths)return; // no light
	PathSegment& seg = pathSegments[path_idx];
	ShadeableIntersection& intersect = intersections[path_idx];
	if (seg.remainingBounces <= 0) // end bounce
	{
		return;
	}
	if (intersect.t <= 0.) // no intersection
	{
		seg.color = glm::vec3(0.f);
		seg.remainingBounces = 0;
		return;
	}
	Material material = materials[intersect.materialId];
#if DEBUG
	seg.color = intersect.surfaceNormal * 0.5f + glm::vec3(0.5);
	return;
#else

	if (material.emittance > 0) // hit light
	{
		seg.color *= (material.color * material.emittance);
		seg.remainingBounces = 0;
		return;
	}
	--seg.remainingBounces;
	if (seg.remainingBounces <= 0) // end bounce and didn't hit light
	{
		seg.color = glm::vec3(0.);
		return;
	}

	float pdf = 1.f;

	seg.ray.origin = intersect.t * seg.ray.direction + seg.ray.origin;

	glm::vec3 wo = -seg.ray.direction;

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_idx, depth);
	if (!sampleRay(wo, intersect.surfaceNormal, material, rng, pdf, seg.ray.direction)) {
		//this is a ray need to be discarded
		seg.remainingBounces = 0;
		seg.color = glm::vec3(0.);
		return;
	}

	//fix strange artifact
	seg.ray.origin += 0.01f * seg.ray.direction;

	glm::vec3 bsdf = getBSDF(seg.ray.direction, wo, intersect.surfaceNormal, material);
	//           albedo           absdot
	seg.color *= (bsdf * glm::clamp(abs(glm::dot(intersect.surfaceNormal, seg.ray.direction)), 0.f, 1.f) / pdf);
	
#endif
}


void PathTracer::pathtrace(uchar4* pbo, int frame, int iter)
{
#if DEBUG
	const int traceDepth = 1;
#else
	const int traceDepth = hst_scene->state.traceDepth;
#endif

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
	PathSegment* dev_paths = this->dev_path.get();
	ShadeableIntersection* dev_intersections = this->dev_intersect.get();
	Mesh* dev_meshs = this->dev_geoms.get();
	Triangle* dev_trigs = this->dev_trigs.get();
	int num_trigs = hst_scene->trigs.size();
	Material* dev_materials = this->dev_mat.get();
	GuiDataContainer* guiData = this->m_guiData;
	glm::vec3* dev_image = this->dev_img.get();
	BVHNode* dev_bvh = this->dev_bvh.get();
	int num_bvh = this->dev_bvh.size();
	
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;

	thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
	thrust::device_ptr<PathSegment> thrust_paths_end(dev_paths + pixelcount);
	int num_paths = thrust_paths_end - thrust_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if BVH
		//int depth
		//	, int num_paths
		//	, PathSegment* in_paths
		//	, Triangle* in_trigs
		//	, Mesh* in_meshs
		//	, int num_trigs
		//	, LinearBVHNode* in_bvh
		//	, int num_bvh
		//	, ShadeableIntersection* out_intersects
		//	)
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_trigs
			, dev_meshs
			, num_trigs
			, dev_bvh
			, num_bvh
			, dev_intersections
			);
#else
		//int depth
		//	, int num_paths
		//	, PathSegment* in_paths
		//	, Triangle* in_trigs
		//	, Mesh* in_meshs
		//	, int num_trigs
		//	, ShadeableIntersection* out_intersects
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_trigs
			, dev_meshs
			, num_trigs
			, dev_intersections
			);
#endif
		checkCUDAError("trace one bounce");



		cudaDeviceSynchronize();
		
		//TODO:
		//--- Shading Stage ---
		//Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		processPBR << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, depth
			, num_paths
			, dev_intersections
			, dev_paths
			, dev_materials
			);

		// * TODO: Stream compact away all of the terminated paths.
		// You may use either your implementation or `thrust::remove_if` or its
		// cousins.
		// * Note that you can't really use a 2D kernel launch any more - switch
		// to 1D.
#if DEBUG
#else
		//thrust_paths_end = thrust::remove_if(thrust_paths, thrust_paths_end, is_black());
		//num_paths = thrust_paths_end - thrust_paths;
#endif
		depth++;
		iterationComplete = depth >= (traceDepth);
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, iter, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_image, iter);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
