#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

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

struct continue_ray
{
	__host__ __device__
		bool operator()(const PathSegment& p)
	{
		return p.remainingBounces != 0;
	}
};

struct sort_on_material_id
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
	{
		return s1.materialId < s2.materialId;
	}
};

__host__ __device__ glm::vec2 sampleDiscConcentric(const glm::vec2& in) {
	glm::vec2 offset = 2.f * in - glm::vec2(1.f, 1.f);
	if (offset.x == 0 && offset.y == 0) {
		return glm::vec2(0.f, 0.f);
	}
	float r, theta;
	if (std::abs(offset.x) > std::abs(offset.y)) {
		r = offset.x;
		theta = PI_OVER_FOUR * (offset.y / offset.x);
	}
	else {
		r = offset.y;
		theta = PI_OVER_TWO - PI_OVER_FOUR * (offset.x / offset.y);
	}
	return r * glm::vec2(cos(theta), sin(theta));
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

		//pix = pix/(pix + glm::vec3(1.f));
		//pix = glm::pow(pix, glm::vec3(1.f/2.2f));

		pix /= iter;
#if REINHARD_GAMMA
		pix /= (pix + glm::vec3(1.0f));
		pix = glm::pow(pix, glm::vec3(1.f/2.2f));
#endif

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);


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
static Triangle* dev_tris = NULL;
static Texture* dev_albedo_textures = NULL;
static glm::vec3* dev_textures = NULL;
static PathSegment* dev_paths = NULL;
#if FIRST_BOUNCE_CACHED
static ShadeableIntersection* dev_intersections_cached = NULL;
#endif
#if USE_BVH
static LBVHNode* dev_lbvh_nodes = NULL;
#endif
static ShadeableIntersection* dev_intersections = NULL;
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

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tris, scene->meshTris.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->meshTris.data(), scene->meshTris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_albedo_textures, scene->albedoTex.size() * sizeof(Texture));
	cudaMemcpy(dev_albedo_textures, scene->albedoTex.data(), scene->albedoTex.size() * sizeof(Texture), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if FIRST_BOUNCE_CACHED
	cudaMalloc(&dev_intersections_cached, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_cached, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if USE_BVH
	buildBVH(scene->root, scene->boundingBoxes);
	int nodes = 0;
	nofOfNodesInBVH(scene->root, nodes);
	std::vector<LBVHNode> flattenedBVH(nodes, LBVHNode());
	int offset = 0;
	flattenBVH(flattenedBVH, scene->root, offset);
	cudaMalloc(&dev_lbvh_nodes, flattenedBVH.size() * sizeof(LBVHNode));
	cudaMemcpy(dev_lbvh_nodes, flattenedBVH.data(), flattenedBVH.size() * sizeof(LBVHNode), cudaMemcpyHostToDevice);
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_tris);
	cudaFree(dev_albedo_textures);
	cudaFree(dev_textures);
	cudaFree(dev_intersections);
#if FIRST_BOUNCE_CACHED
	cudaFree(dev_intersections_cached);
#endif
#if USE_BVH
	cudaFree(dev_lbvh_nodes);
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
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.accumCol = glm::vec3(1.0f, 1.0f, 1.0f);

		float dX = 0, dY = 0;

#if ANTI_ALIASING || DEPTH_OF_FIELD
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> uX(0, 1);
		thrust::uniform_real_distribution<float> uY(0, 1);
#endif

#if ANTI_ALIASING			
		dX = uX(rng); dY = uY(rng);
#endif

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)(x + dX) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)(y + dY) - (float)cam.resolution.y * 0.5f)
		);

#if DEPTH_OF_FIELD
		//Entirely referenced PBRT : https://pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField

		glm::vec2 pointOnLens = cam.lensRadius * sampleDiscConcentric(glm::vec2(uX(rng), uY(rng)));

		//How I got this t-value?
		//Eq 1: cam.pos + t * ray.dir = pointOnFilm
		//Eq 2: (pointOnFilm - (cam.pos + focalLength * view)).(view) = 0. Ray-plane intersection: dot product of a vector on the plane with its normal = 0
		//Solve for t. PBRT equation (presumably) assumes view vector to align with the z axis and the lens to be at the origin
		float t = cam.focalLength * glm::dot(cam.view, cam.view)/glm::dot(segment.ray.direction, cam.view);
		glm::vec3 pointOnFilm = getPointOnRay(segment.ray, t);

		segment.ray.origin += glm::vec3(pointOnLens.x, pointOnLens.y, 0);
		segment.ray.direction = glm::normalize(pointOnFilm - segment.ray.origin);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, Triangle* tris
#if USE_BVH
	, LBVHNode* lbvh
#endif
	, int geoms_size
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
		glm::vec2 uv;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

#if USE_BVH
		//BVH traversal
#if 0
		int currNodeIdx = 0, visitOffset = 0;
		int stack[64] = { 0 };
		while (true) {
			LBVHNode* currNode = &lbvh[currNodeIdx];
			if (doesRayIntersectAABB(pathSegment.ray, currNode->boundingBox, tris, currNode->isLeaf, t, tmp_intersect, tmp_normal)) {
				if (currNode->isLeaf) {

					Geom geom = currNode->boundingBox.geom;

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
						t = rayTriangleIntersection(geom, pathSegment.ray, tris, currNode->boundingBox.triIdx, tmp_intersect, tmp_normal);
						//t = meshGltfIntersectionTest(geom, pathSegment.ray, tris, tmp_intersect, tmp_normal);
					}

					if (t < t_min) {
						t_min = t;
						hit_geom_index = currNode->boundingBox.geom.geomId;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
					}
					if (visitOffset == 0) break;
					currNodeIdx = stack[--visitOffset];
				}
				else {
					/*stack[visitOffset++] = currNode->secondChildOffset;
					currNodeIdx += 1;*/
					stack[visitOffset++] = currNodeIdx + 1;
					currNodeIdx = currNode->secondChildOffset;
				}
			}
			else {
				if (visitOffset == 0) break;
				currNodeIdx = stack[--visitOffset];
			}
		}
#endif

#if 1
		int top = 0;
		int stack[64] = { 0 };
		while (top >= 0 && top < 64) {
			int currNodeIdx = stack[top];
			top--;

			LBVHNode* currNode = &lbvh[currNodeIdx];
			//if (doesRayIntersectAABB(pathSegment.ray, currNode->boundingBox)) {
				if (currNode->isLeaf) {
					
					Geom geom = currNode->boundingBox.geom;
					
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
						t = rayTriangleIntersection(geom, pathSegment.ray, tris, currNode->boundingBox.triIdx, tmp_intersect, tmp_normal, tmp_uv);
					}

					// Compute the minimum t from the intersection tests to determine what
					// scene geometry object was hit first.
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						hit_geom_index = currNode->boundingBox.geom.geomId;
						intersect_point = tmp_intersect;
						normal = tmp_normal;						
						uv = tmp_uv;
					}
				}
				else {
					if (doesRayIntersectAABB(pathSegment.ray, lbvh[currNodeIdx + 1].boundingBox)) {
						top++;
						stack[top] = currNodeIdx + 1; //left child
					}

					if (currNode->secondChildOffset != -1 && doesRayIntersectAABB(pathSegment.ray, lbvh[currNode->secondChildOffset].boundingBox)) {
						top++;
						stack[top] = currNode->secondChildOffset; //right child
					}
				}
			//}
		}
#endif
		
#else
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
				t = meshGltfIntersectionTest(geom, pathSegment.ray, tris, tmp_intersect, tmp_normal);
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
#endif

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			pathSegment.remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].hasUV = geoms[hit_geom_index].hasAlbedoMap;
			intersections[path_index].uv = uv;
			intersections[path_index].texId = geoms[hit_geom_index].albedoTexId;
		}
	}
}

__global__ void kernShadeAllMaterials(
	int iter
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Texture* albedoTexMetadata
	, glm::vec3* textureCols
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0)
			return;
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];			

			if (intersection.hasUV) {
				int X = glm::min(albedoTexMetadata[intersection.texId].width * intersection.uv.x, albedoTexMetadata[intersection.texId].width - 1.0f);
				int Y = glm::min(albedoTexMetadata[intersection.texId].height * (intersection.uv.y), albedoTexMetadata[intersection.texId].height - 1.0f);
				int idx = albedoTexMetadata[intersection.texId].width * Y + X + albedoTexMetadata[intersection.texId].startIdx;
				material.color = textureCols[idx];
			}
			glm::vec3 materialColor = material.color;
			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color = (materialColor * material.emittance) * pathSegments[idx].accumCol;
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				scatterRay(pathSegments[idx],
					getPointOnRay(pathSegments[idx].ray, intersection.t),
					intersection.surfaceNormal,
					material,
					rng);				
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
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
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if FIRST_BOUNCE_CACHED
		if (iter == 1 && depth == 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_tris
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace first cached bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections_cached, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersections_cached, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
#endif
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, dev_tris
#if USE_BVH
				, dev_lbvh_nodes
#endif
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
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

#if MATERIAL_SORT
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sort_on_material_id());
#endif
		kernShadeAllMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_albedo_textures,
			dev_textures
			);

		cudaDeviceSynchronize();

#if STREAM_COMPACTION
		thrust::device_ptr<PathSegment> thrust_dev_paths(dev_paths);
		thrust::device_ptr<PathSegment> end = thrust::stable_partition(thrust::device, thrust_dev_paths, thrust_dev_paths + num_paths, continue_ray());
		num_paths = thrust::distance(thrust_dev_paths, end);
#endif

		iterationComplete = (depth == traceDepth) || (num_paths == 0);

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
