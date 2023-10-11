#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <curand_kernel.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define BlockSize 128
#define STREAM_COMPACTION 1
#define SORT_MATERIAL 1
#define CACHE_FIRST_BOUNCE 1
#define ANTI_ALIASING 0
#define DIRECT_LIGHT 0
#define BVHOPEN 1

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
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection* dev_cache_intersections = NULL;
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texColors = NULL;
static Geom* lights = NULL;
static BVHNode_GPU* dev_bvh_nodes = NULL;

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
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_texColors, scene->textureColors.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_texColors, scene->textureColors.data(), scene->textureColors.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&lights, scene->lightNum * sizeof(Geom));
	cudaMemcpy(lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_nodes, scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU));
	cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes_gpu.data(), scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cache_intersections);
	cudaFree(dev_triangles);
	cudaFree(dev_textures);
	cudaFree(dev_texColors);
	cudaFree(lights);
	cudaFree(dev_bvh_nodes);
	checkCUDAError("pathtraceFree");
}

#define M_PI 3.14159265358979323846
__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	if (uOffset.x == 0.f && uOffset.y == 0.f)
		return glm::vec2(0.f);

	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = (M_PI / 4.f) * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = (M_PI / 2.f) - (M_PI / 4.f) * (uOffset.x / uOffset.y);
	}

	return r * glm::vec2(std::cos(theta), std::sin(theta));
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
		// Antialiasing by jittering the ray
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
		float randomX = u01(rng);
		float randomY = u01(rng);
		//Stochastic Sampled Antialiasing
#if     ANTI_ALIASING
		float jitterAmount = 0.5f;
		float jitterX = (randomX - 0.5f) * jitterAmount;
		float jitterY = (randomY - 0.5f) * jitterAmount;
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

		// Depth of Field effect
		// Ref: PBTR 6.2
		if (cam.aperture > 0.0f) {
			float lensRadius = cam.aperture / 2.0f;
			glm::vec2 randomCircle = ConcentricSampleDisk(glm::vec2(randomX, randomY));
			glm::vec2 pointOnLens = lensRadius * randomCircle;

			float focalDistance = length(cam.lookAt - cam.position) * lensRadius / cam.aperture;
			glm::vec3 focalPoint = segment.ray.origin + focalDistance * segment.ray.direction;

			// Update ray's origin and direction for Depth of Field
			segment.ray.origin += cam.right * pointOnLens.x + cam.up * pointOnLens.y;
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
		}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__host__ __device__ bool checkMeshhBoundingBox(Geom& geom, Ray& ray) {
	Ray q;
	q.origin = glm::vec3(geom.inverseTransform * glm::vec4(ray.origin, 1.0f));
	q.direction = glm::normalize(glm::vec3(geom.inverseTransform * glm::vec4(ray.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;

	for (int xyz = 0; xyz < 3; ++xyz) {
		if (glm::abs(q.direction[xyz]) > 0.00001f) {
			float t1 = (geom.boundingBoxMin[xyz] - q.origin[xyz]) / q.direction[xyz];
			float t2 = (geom.boundingBoxMax[xyz] - q.origin[xyz]) / q.direction[xyz];
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));
		}
	}

	return tmax >= tmin && tmax > 0;
}

__host__ __device__ bool intersectBoundingBox(const glm::vec3& minCorner, const glm::vec3& maxCorner, const Ray& ray) {
	float tmin = (minCorner.x - ray.origin.x) / ray.direction.x;
	float tmax = (maxCorner.x - ray.origin.x) / ray.direction.x;

	if (tmin > tmax) std::swap(tmin, tmax);

	float tymin = (minCorner.y - ray.origin.y) / ray.direction.y;
	float tymax = (maxCorner.y - ray.origin.y) / ray.direction.y;

	if (tymin > tymax) std::swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (minCorner.z - ray.origin.z) / ray.direction.z;
	float tzmax = (maxCorner.z - ray.origin.z) / ray.direction.z;

	if (tzmin > tzmax) std::swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin < 0 && tmax < 0)
		return false;

	return true;
}


//__host__ __device__ int intersectOctree(int nodeIndex, const Ray& ray, Triangle* triangles, Geom* potentialIntersections, int& count) {
//	if (nodeIndex < 0 || !intersectBoundingBox(dev_nodes[nodeIndex].minCorner, dev_nodes[nodeIndex].maxCorner, ray)) {
//		return count;
//	}
//
//	OctreeNode* node = &dev_nodes[nodeIndex];
//
//	for (int j = 0; j < node->objects.size(); j++) {
//		const Geom& obj = node->objects[j];
//		bool intersects = false;
//		glm::vec3 tmp_intersect, tmp_normal;
//		glm::vec2 tmp_uv;
//		bool outside = true;  // Assuming you have this from your previous context
//
//		switch (obj.type) {
//		case CUBE:
//			intersects = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside) > 0.0f;
//			break;
//		case SPHERE:
//			intersects = sphereIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside) > 0.0f;
//			break;
//		case OBJMESH:
//			intersects = triangleIntersectionTest(obj, ray, tmp_intersect, triangles, obj.triangleIdStart, obj.triangleIdEnd, tmp_normal, outside, tmp_uv) > 0.0f;
//			break;
//		default:
//			break;
//		}
//
//		if (intersects) {
//			potentialIntersections[count] = obj;
//			count++;
//		}
//	}
//
//	for (int i = 0; i < 8; ++i) {
//		count = intersectOctree(node->children[i], ray, triangles, potentialIntersections, count);
//	}
//
//	return count;
//}




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
	, Triangle* triangles
	, BVHNode_GPU* bvh_nodes
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
		glm::vec2 tmp_uv;

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
			else if (geom.type == OBJMESH) {
				if (!checkMeshhBoundingBox(geom, pathSegment.ray))
				{
					break;
				}
				float z = FLT_MAX;

				for (int j = geom.triangleIdStart; j <= geom.triangleIdEnd; j++) {
					Triangle& tri = triangles[j];

					glm::vec3 cur_intersect;
					glm::vec3 cur_normal;
					glm::vec2 cur_uv;
					bool cur_outside;

					float tempT = triangleIntersectionTest(geom, pathSegment.ray, cur_intersect, tri, cur_normal, cur_uv, cur_outside);

					if (tempT < z) {
						z = tempT;
						tmp_intersect = cur_intersect;
						tmp_normal = cur_normal;
						tmp_uv = cur_uv;
						outside = cur_outside;
					}
				}
				t = z;

			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.

			if (t > EPSILON && t_min > t)
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
			pathSegments[path_index].remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = tmp_uv;
			intersections[path_index].textureId = geoms[hit_geom_index].textureId;
			intersections[path_index].bumpTextureId = geoms[hit_geom_index].bumpTextureId;
		}
	}
}

__global__ void computeBVHIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles
	, BVHNode_GPU* bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		Ray r = pathSegment.ray;
		r.direction_inv = 1.f / r.direction;  // Compute inverse direction upfront

		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		int matId = 1;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
		bool outside = true;

		int stack_pointer = 0;
		int cur_node_index = 0;
		int node_stack[32];
		BVHNode_GPU cur_node;

		while (true)
		{
			cur_node = bvh_nodes[cur_node_index];

			float t1, t2, tmin, tmax;

			t1 = (cur_node.AABB_min.x - r.origin.x) * r.direction_inv.x;
			t2 = (cur_node.AABB_max.x - r.origin.x) * r.direction_inv.x;
			tmin = glm::min(t1, t2);
			tmax = glm::max(t1, t2);

			t1 = (cur_node.AABB_min.y - r.origin.y) * r.direction_inv.y;
			t2 = (cur_node.AABB_max.y - r.origin.y) * r.direction_inv.y;
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));

			t1 = (cur_node.AABB_min.z - r.origin.z) * r.direction_inv.z;
			t2 = (cur_node.AABB_max.z - r.origin.z) * r.direction_inv.z;
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));

			if (tmax >= tmin && t_min > tmin)
			{
				if (cur_node.tri_index != -1)
				{
					Triangle& tri = triangles[cur_node.tri_index];
					float t = glm::dot(tri.plane_normal, (tri.vertices[0] - r.origin)) / glm::dot(tri.plane_normal, r.direction);

					if (t >= -0.0001f && t_min > t)
					{
						glm::vec3 P = r.origin + t * r.direction;
						glm::vec3 s = glm::vec3(
							glm::length(glm::cross(P - tri.vertices[1], P - tri.vertices[2])),
							glm::length(glm::cross(P - tri.vertices[2], P - tri.vertices[0])),
							glm::length(glm::cross(P - tri.vertices[0], P - tri.vertices[1]))
						) / tri.S;

						if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
							s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f))
						{
							hit_geom_index = geoms_size + 1;
							t_min = t;
							normal = glm::normalize(s.x * tri.normals[0] + s.y * tri.normals[1] + s.z * tri.normals[2]);
							matId = tri.mat_ID;
						}
					}

					if (stack_pointer == 0)
					{
						break;
					}
					stack_pointer--;
					cur_node_index = node_stack[stack_pointer];
				}
				else
				{
					node_stack[stack_pointer] = cur_node.offset_to_second_child;
					stack_pointer++;
					cur_node_index++;
				}
			}
			else
			{
				if (stack_pointer == 0)
				{
					break;
				}
				stack_pointer--;
				cur_node_index = node_stack[stack_pointer];
			}
		}

		// Only traverse the geoms if BVH traversal yielded no results
		if (hit_geom_index == -1)
		{
			for (int i = 0; i < geoms_size; i++)
			{
				Geom& geom = geoms[i];
				float t;
				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				}

				if (t > EPSILON && t_min > t)
				{
					t_min = t;
					hit_geom_index = i;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
				}
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			pathSegments[path_index].remainingBounces = 0;
		}
		else
		{
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = (hit_geom_index > geoms_size) ? matId : geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = tmp_uv;
			intersections[path_index].textureId = -1;
			intersections[path_index].bumpTextureId = -1;
		}
	}



}

float getHeightValue(Texture& tex, float u, float v) {
	int w = tex.width * u;
	int h = tex.height * (1 - v);
	int colIdx = h * tex.width + w;

	return tex.bumpMap[colIdx];
}

__host__ __device__ glm::vec3 getBumpedNormal(Texture& tex, glm::vec2 uv) {
	float delta = 1.0f / tex.width;
	float heightCenter = getHeightValue(tex, uv.x, uv.y);
	float heightU = getHeightValue(tex, uv.x + delta, uv.y);
	float heightV = getHeightValue(tex, uv.x, uv.y + delta);

	glm::vec3 tangent = glm::vec3(1, 0, heightU - heightCenter);
	glm::vec3 bitangent = glm::vec3(0, 1, heightV - heightCenter);

	return glm::cross(tangent, bitangent);
}


__global__ void shadeBSDFMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* lights
	, int lightNum
	, Texture* textures
	, glm::vec3* texColors
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	PathSegment& path = pathSegments[idx];

	if (idx < num_paths && pathSegments[idx].remainingBounces >= 0)
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

			/*if (intersection.textureId != -1 && intersection.bumpTextureId != -1) {
				Texture& tex = textures[intersection.textureId];
				glm::vec3 bumpedNormal = getBumpedNormal(tex, intersection.uv);
				intersection.surfaceNormal = glm::normalize(bumpedNormal);
			}
			else if (intersection.textureId != -1)*/
			if (intersection.textureId != -1)
			{
				Texture tex = textures[intersection.textureId];
				int w = tex.width * intersection.uv[0] - 0.5;
				int h = tex.height * (1 - intersection.uv[1]) - 0.5;
				int colIdx = h * tex.width + w + tex.idx;
				material.color = texColors[colIdx];
			}

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				path.color *= (materialColor * material.emittance);
				path.remainingBounces = 0;
				return;
			}
			else {
				glm::vec3 intersectionPoint = glm::normalize(path.ray.direction) * intersection.t + path.ray.origin;
				scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng, lights, lightNum);

#if DIRECT_LIGHT
				//Simple version of direct lighting
				if (path.remainingBounces == 2)
				{
					thrust::uniform_real_distribution<float> u01(0, 1);
					thrust::uniform_real_distribution<int> u02(0, lightNum - 1);
					Geom light = lights[u02(rng)];
					glm::vec3 sampledLight = glm::vec3(light.transform * glm::vec4(u01(rng), u01(rng), u01(rng), 1.f));
					path.ray.direction = glm::normalize(sampledLight - path.ray.origin);
					return;
				}
#endif
				//path.remainingBounces--;
			}
		}
		else {
			path.color = glm::vec3(0.0f);
			path.remainingBounces = 0;
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

struct isPathCompleted
{
	__host__ __device__ bool operator()(const PathSegment& pathSegment) {
		return pathSegment.remainingBounces > 0;
	}
};

struct dev_materialIds
{
	__host__ __device__ bool operator()(const ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2)
	{
		return intersect1.materialId < intersect2.materialId;
	}
};

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
	int oriPaths = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if	CACHE_FIRST_BOUNCE && ! ANTI_ALIASING
		if (iter > 1) {
			cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		
#endif
		computeBVHIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_triangles
			, dev_bvh_nodes
			);

#if CACHE_FIRST_BOUNCE && ! ANTI_ALIASING
		if (iter == 1)
		{
			cudaMemcpy(dev_cache_intersections, dev_intersections,
				pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
#if SORT_MATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, dev_materialIds());
#endif
		shadeBSDFMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			lights,
			hst_scene->lightNum,
			dev_textures,
			dev_texColors
			);

#if STREAM_COMPACTION
		PathSegment* path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, isPathCompleted());
		num_paths = path_end - dev_paths;
		iterationComplete = (num_paths == 0);
#endif
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (oriPaths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
