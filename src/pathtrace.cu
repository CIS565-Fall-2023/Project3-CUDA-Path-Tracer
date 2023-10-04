#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

__device__ inline bool util_math_is_nan(const glm::vec3& v)
{
	return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}



//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
#if TONEMAPPING
		color = pix / (float)iter;
		color = util_postprocess_gamma(util_postprocess_ACESFilm(color));
		color = color * 255.0f;
#else
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif
		if (util_math_is_nan(pix))
		{
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else
		{
			// Each thread writes one pixel location in the texture (textel)
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
		pbo[index].w = 0;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Object* dev_objs = NULL;
static Material* dev_materials = NULL;
static BVHGPUNode* dev_bvhArray = NULL;
static MTBVHGPUNode* dev_mtbvhArray = NULL;
static Primitive* dev_primitives = NULL;
static glm::ivec3* dev_triangles = NULL;
static glm::vec3* dev_vertices = NULL;
static glm::vec2* dev_uvs = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec3* dev_tangents = NULL;
static float* dev_fsigns = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
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

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_objs, scene->objects.size() * sizeof(Object));
	cudaMemcpy(dev_objs, scene->objects.data(), scene->objects.size() * sizeof(Object), cudaMemcpyHostToDevice);

	if (scene->triangles.size())
	{
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(glm::ivec3));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_vertices, scene->verticies.size() * sizeof(glm::vec3));
		cudaMemcpy(dev_vertices, scene->verticies.data(), scene->verticies.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_uvs, scene->uvs.size() * sizeof(glm::vec2));
		cudaMemcpy(dev_uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
		if (scene->normals.size())
		{
			cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->tangents.size())
		{
			cudaMalloc(&dev_tangents, scene->tangents.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_tangents, scene->tangents.data(), scene->tangents.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->fSigns.size())
		{
			cudaMalloc(&dev_fsigns, scene->fSigns.size() * sizeof(float));
			cudaMemcpy(dev_fsigns, scene->fSigns.data(), scene->fSigns.size() * sizeof(float), cudaMemcpyHostToDevice);
		}
	}
#if MTBVH
	cudaMalloc(&dev_mtbvhArray, scene->MTBVHArray.size() * sizeof(MTBVHGPUNode));
	cudaMemcpy(dev_mtbvhArray, scene->MTBVHArray.data(), scene->MTBVHArray.size() * sizeof(MTBVHGPUNode), cudaMemcpyHostToDevice);
#else
	cudaMalloc(&dev_bvhArray, scene->bvhArray.size() * sizeof(BVHGPUNode));
	cudaMemcpy(dev_bvhArray, scene->bvhArray.data(), scene->bvhArray.size() * sizeof(BVHGPUNode), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
	cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree(Scene* scene) {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths1);
	cudaFree(dev_paths2);
	cudaFree(dev_objs);
	if (scene->triangles.size())
	{
		cudaFree(dev_triangles);
		cudaFree(dev_vertices);
		cudaFree(dev_uvs);
		if (scene->normals.size())
		{
			cudaFree(dev_normals);
		}
		if (scene->tangents.size())
		{
			cudaFree(dev_tangents);
		}
		if (scene->fSigns.size())
		{
			cudaFree(dev_fsigns);
		}
	}
	cudaFree(dev_primitives);
#if MTBVH
	cudaFree(dev_mtbvhArray);
#else
	cudaFree(dev_bvhArray);
#endif
	cudaFree(dev_materials);
	cudaFree(dev_intersections1);
	cudaFree(dev_intersections2);
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

__device__ inline glm::vec2 util_concentric_sample_disk(glm::vec2 rand)
{
	rand = 2.0f * rand - 1.0f;
	if (rand.x == 0 && rand.y == 0)
	{
		return glm::vec2(0);
	}
	const float pi_4 = PI / 4, pi_2 = PI / 2;
	bool x_g_y = abs(rand.x) > abs(rand.y);
	float theta = x_g_y ? pi_4 * rand.y / rand.x : pi_2 - pi_4 * rand.x / rand.y;
	float r = x_g_y ? rand.x : rand.y;
	return glm::vec2(cos(theta), sin(theta)) * r;
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
	thrust::default_random_engine rng = makeSeededRandomEngine(x, y, iter);
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#if STOCHASTIC_SAMPLING
		// TODO: implement antialiasing by jittering the ray
		glm::vec2 jitter = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter[0])
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter[1])
		);
#if DOF_ENABLED
		float lensR = cam.lensRadius;
		glm::vec3 perpDir = glm::cross(cam.right, cam.up);
		perpDir = glm::normalize(perpDir);
		float focalLen = cam.focalLength;
		float tFocus = focalLen / glm::abs(glm::dot(segment.ray.direction, perpDir));
		glm::vec2 offset = lensR * util_concentric_sample_disk(glm::vec2(u01(rng), u01(rng)));
		glm::vec3 newOri = offset.x * cam.right + offset.y * cam.up + cam.position;
		glm::vec3 pFocus = segment.ray.direction * tFocus + segment.ray.origin;
		segment.ray.direction = glm::normalize(pFocus - newOri);
		segment.ray.origin = newOri;
#endif

#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void compute_intersection(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Object* geoms
	, int objs_size
	, glm::ivec3* modelTriangles
	, glm::vec3* modelVertices
	, const glm::vec2* modelUVs
	, const glm::vec3* modelNormals
	, cudaTextureObject_t skyboxTex
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment& pathSegment = pathSegments[path_index];
		float t = -1.0;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_material_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < objs_size; i++)
		{
			Object& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
			}
			else if (geom.type == SPHERE)
			{
				t = util_geometry_ray_sphere_intersection(geom, pathSegment.ray, tmp_intersect, tmp_normal);
			}
			else if (geom.type == TRIANGLE_MESH)
			{
				glm::vec3 baryCoord;
				for (int i = geom.triangleStart; i != geom.triangleEnd; i++)
				{
					const glm::ivec3& tri = modelTriangles[i];
					const glm::vec3& v0 = modelVertices[tri[0]];
					const glm::vec3& v1 = modelVertices[tri[1]];
					const glm::vec3& v2 = modelVertices[tri[2]];
					t = triangleIntersectionTest(geom.Transform, v0, v1, v2, pathSegment.ray, tmp_intersect, tmp_normal, baryCoord);
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						hit_material_index = geom.materialid;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
					}
				}
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_material_index = geom.materialid;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (t_min == FLT_MAX)//hits nothing
		{
			rayValid[path_index] = 0;
			if (skyboxTex)
			{
				glm::vec2 uv = util_sample_spherical_map(glm::normalize(pathSegment.ray.direction));
				float4 skyColorRGBA = tex2D<float4>(skyboxTex, uv.x, uv.y);
				glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
				image[pathSegment.pixelIndex] += pathSegment.color * skyColor;
			}
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_material_index;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].worldPos = intersect_point;
			rayValid[path_index] = 1;
		}
	}
}

__device__ inline int util_bvh_get_sibling(const BVHGPUNode* bvhArray, int curr)
{
	int parent = bvhArray[curr].parent;
	if (parent == -1) return -1;
	return bvhArray[parent].left == curr ? bvhArray[parent].right : bvhArray[parent].left;
}

__device__ inline int util_bvh_get_near_child(const BVHGPUNode* bvhArray, int curr, const glm::vec3& rayDir)
{
	return rayDir[bvhArray[curr].axis] > 0.0 ? bvhArray[curr].left : bvhArray[curr].right;
}

__device__ inline bool util_bvh_is_leaf(const BVHGPUNode* bvhArray, int curr)
{
	return bvhArray[curr].left == -1 && bvhArray[curr].right == -1;
}

__device__ inline bool util_bvh_leaf_intersect(const Primitive* primitives, 
	int primsStart, 
	int primsEnd, 
	const Object* objects, 
	const glm::ivec3* modelTriangles, 
	const  glm::vec3* modelVertices, 
	const  glm::vec2* modelUVs,
	const glm::vec3* modelNormals, 
	const glm::vec3* modelTangents,
	const float* modelFsigns,
	const Ray& ray, 
	ShadeableIntersection* intersection
)
{
	glm::vec3 tmp_intersect, tmp_normal, tmp_baryCoord, tmp_tangent;
	float tmp_fsign;
	glm::vec2	tmp_uv;
	bool intersected = false;
	float t = -1.0;
	for (int i = primsStart; i != primsEnd; i++)
	{
		const Primitive& prim = primitives[i];
		int objID = prim.objID;
		const Object& obj = objects[objID];
		
		if (obj.type == TRIANGLE_MESH)
		{
			const glm::ivec3& tri = modelTriangles[obj.triangleStart + prim.offset];
			const glm::vec3& v0 = modelVertices[tri[0]];
			const glm::vec3& v1 = modelVertices[tri[1]];
			const glm::vec3& v2 = modelVertices[tri[2]];
			t = triangleIntersectionTest(obj.Transform, v0, v1, v2, ray, tmp_intersect, tmp_normal, tmp_baryCoord);
			if (modelNormals && modelUVs)
			{
				const glm::vec3& n0 = modelNormals[tri[0]];
				const glm::vec3& n1 = modelNormals[tri[1]];
				const glm::vec3& n2 = modelNormals[tri[2]];
				tmp_normal = n0 * tmp_baryCoord[0] + n1 * tmp_baryCoord[1] + n2 * tmp_baryCoord[2];
				tmp_normal = glm::vec3(obj.Transform.invTranspose * glm::vec4(tmp_normal, 0.0));//TODO: precompute transformation
				const glm::vec2& uv0 = modelUVs[tri[0]];
				const glm::vec2& uv1 = modelUVs[tri[1]];
				const glm::vec2& uv2 = modelUVs[tri[2]];
				tmp_uv = uv0 * tmp_baryCoord[0] + uv1 * tmp_baryCoord[1] + uv2 * tmp_baryCoord[2];
				const glm::vec3& t0 = modelTangents[tri[0]];
				const glm::vec3& t1 = modelTangents[tri[1]];
				const glm::vec3& t2 = modelTangents[tri[2]];
				tmp_tangent = t0 * tmp_baryCoord[0] + t1 * tmp_baryCoord[1] + t2 * tmp_baryCoord[2];
				tmp_tangent = glm::vec3(obj.Transform.invTranspose * glm::vec4(tmp_tangent, 0.0));
			}
		}
		else if (obj.type == CUBE)
		{
			t = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal);
		}
		else if (obj.type == SPHERE)
		{
			t = util_geometry_ray_sphere_intersection(obj, ray, tmp_intersect, tmp_normal);
		}
		
		if (t > 0.0 && t < intersection->t)
		{
			intersection->t = t;
			intersection->materialId = obj.materialid;
			intersection->worldPos = tmp_intersect;
			intersection->surfaceNormal = tmp_normal;
			intersection->surfaceTangent = tmp_tangent;
			intersection->fsign = tmp_fsign;
			intersection->uv = tmp_uv;
			intersected = true;
		}
		
	}
	return intersected;
}

enum bvh_traverse_state {
	fromChild,fromParent,fromSibling
};

__global__ void compute_intersection_bvh_stackless(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Material* materials
	, const Object* objects
	, int objs_size
	, const glm::ivec3* modelTriangles
	, const glm::vec3* modelVertices
	, const glm::vec2* modelUVs
	, const glm::vec3* modelNormals
	, const glm::vec3* modelTangents
	, const float* modelFsigns
	, cudaTextureObject_t skyboxTex
	, const Primitive* primitives
	, const BVHGPUNode* bvhArray
	, int bvhArraySize
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	int curr = util_bvh_get_near_child(bvhArray, 0, rayOri);
	bvh_traverse_state state = fromParent;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = 1e37f;
	bool intersected = false;
	while (curr >= 0 && curr < bvhArraySize)
	{
		if (state == fromChild)
		{
			if (curr == 0) break;
			int parent = bvhArray[curr].parent;
			if (curr == util_bvh_get_near_child(bvhArray, parent, rayOri))
			{
				curr = util_bvh_get_sibling(bvhArray, curr);
				state = fromSibling;
			}
			else
			{
				curr = parent;
				state = fromChild;
			}
		}
		else if (state == fromSibling)
		{
			bool outside = true;
			float boxt = boundingBoxIntersectionTest(bvhArray[curr].bbox, pathSegment.ray, outside);
			if (!outside) boxt = EPSILON;
			if (!(boxt > 0 && boxt < tmpIntersection.t))
			{
				curr = bvhArray[curr].parent;
				state = fromChild;
			}
			else if (util_bvh_is_leaf(bvhArray, curr))
			{
				int start = bvhArray[curr].startPrim, end = bvhArray[curr].endPrim;
				if (util_bvh_leaf_intersect(primitives, start, end, objects, modelTriangles, modelVertices, modelUVs, modelNormals, modelTangents, modelFsigns, pathSegment.ray, &tmpIntersection))
				{
					intersected = true;
				}
				curr = bvhArray[curr].parent;
				state = fromChild;
			}
			else
			{
				curr = util_bvh_get_near_child(bvhArray, curr, rayOri);
				state = fromParent;
			}
		}
		else// from parent
		{
			bool outside = true;
			float boxt = boundingBoxIntersectionTest(bvhArray[curr].bbox, pathSegment.ray, outside);
			if (!outside) boxt = EPSILON;
			if (!(boxt > 0 && boxt < tmpIntersection.t))
			{
				curr = util_bvh_get_sibling(bvhArray, curr);
				state = fromSibling;
			}
			else if (util_bvh_is_leaf(bvhArray, curr))
			{
				int start = bvhArray[curr].startPrim, end = bvhArray[curr].endPrim;
				if (util_bvh_leaf_intersect(primitives, start, end, objects, modelTriangles, modelVertices, modelUVs, modelNormals, modelTangents, modelFsigns, pathSegment.ray, &tmpIntersection))
				{
					intersected = true;
				}
				curr = util_bvh_get_sibling(bvhArray, curr);
				state = fromSibling;
			}
			else
			{
				curr = util_bvh_get_near_child(bvhArray, curr, pathSegment.ray.origin);
				state = fromParent;
			}
		}
	}
	rayValid[path_index] = intersected;
	if (intersected)
	{
		intersections[path_index] = tmpIntersection;
		intersections[path_index].type = materials[tmpIntersection.materialId].type;
	}
	else if(skyboxTex)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(skyboxTex, uv.x, uv.y);
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
		image[pathSegment.pixelIndex] += pathSegment.color * skyColor * BACKGROUND_COLOR_MULT;
	}
}



__global__ void compute_intersection_bvh_stackless_mtbvh(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Material* materials
	, const Object* objects
	, int objs_size
	, const glm::ivec3* modelTriangles
	, const glm::vec3* modelVertices
	, const glm::vec2* modelUVs
	, const glm::vec3* modelNormals
	, const glm::vec3* modelTangents
	, const float* modelFsigns
	, cudaTextureObject_t skyboxTex
	, const Primitive* primitives
	, const MTBVHGPUNode* bvhArray
	, int bvhArraySize
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	Ray& ray = pathSegment.ray;
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	float x = fabs(rayDir.x), y = fabs(rayDir.y), z = fabs(rayDir.z);
	int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
	int sgn = rayDir[axis] > 0 ? 0 : 1;
	int d = (axis << 1) + sgn;
	const MTBVHGPUNode* currArray = bvhArray + d * bvhArraySize;
	int curr = 0;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = 1e37f;
	bool intersected = false;
	while (curr >= 0 && curr < bvhArraySize)
	{
		bool outside = true;
		float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
		if (!outside) boxt = EPSILON;
		if (boxt > 0 && boxt < tmpIntersection.t)
		{
			if (currArray[curr].startPrim != -1)//leaf node
			{
				int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
				bool intersect = util_bvh_leaf_intersect(primitives, start, end, objects, modelTriangles, modelVertices, modelUVs, modelNormals, modelTangents, modelFsigns, ray, &tmpIntersection);
				intersected = intersected || intersect;
			}
			curr = currArray[curr].hitLink;
		}
		else
		{
			curr = currArray[curr].missLink;
		}
	}
	
	rayValid[path_index] = intersected;
	if (intersected)
	{
		intersections[path_index] = tmpIntersection;
		intersections[path_index].type = materials[tmpIntersection.materialId].type;
		pathSegment.remainingBounces--;
	}
	else if (skyboxTex)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(skyboxTex, uv.x, uv.y);
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
		image[pathSegment.pixelIndex] += depth > 0 ? pathSegment.color * skyColor * BACKGROUND_COLOR_MULT : pathSegment.color * skyColor;
	}
}


__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int* rayValid
	, glm::vec3* image
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	Material material = materials[intersection.materialId];
	glm::vec3 materialColor = material.color;
#if VIS_NORMAL
	image[pathSegments[idx].pixelIndex] += (glm::normalize(intersection.surfaceNormal));
	rayValid[idx] = 0;
	return;
#endif

	// If the material indicates that the object was a light, "light" the ray
	if (material.type == MaterialType::emitting) {
		pathSegments[idx].color *= (materialColor * material.emittance);
		rayValid[idx] = 0;
		if (!util_math_is_nan(pathSegments[idx].color))
			image[pathSegments[idx].pixelIndex] += pathSegments[idx].color;
	}
	else {
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 nMap = glm::vec3(0, 0, 1);
		if (material.normalMap != 0)
		{
			float4 nMapCol = tex2D<float4>(material.normalMap, intersection.uv.x, 1.0 - intersection.uv.y);
			nMap.x = nMapCol.x;
			nMap.y = nMapCol.y;
			nMap.z = nMapCol.z;
			nMap=glm::pow(nMap, glm::vec3(1 / 2.2f));
			nMap = nMap * 2.0f - 1.0f;
			nMap = glm::normalize(nMap);
		}
		
		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		glm::vec3 B, T;
		if (material.normalMap != 0)
		{
			T = intersection.surfaceTangent;
			T = glm::normalize(T - N * glm::dot(N, T));
			B = glm::cross(N, T);
			N = glm::normalize(T * nMap.x + B * nMap.y + N * nMap.z);
		}
		else
		{
			util_math_get_TBN_pixar(N, &T, &B);
		}
		glm::mat3 TBN(T, B, N);
		glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi, bxdf;
		glm::vec3 random = glm::vec3(u01(rng), u01(rng), u01(rng));
		if (material.type == MaterialType::metallicWorkflow)
		{
			float4 color = { 0,0,0,1 };
			float roughness = material.roughness, metallic = material.metallic;
			if (material.baseColorMap != 0)
			{
				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, 1.0 - intersection.uv.y);
				materialColor.x = color.x;
				materialColor.y = color.y;
				materialColor.z = color.z;
			}
			if (material.metallicRoughnessMap != 0)
			{
				color = tex2D<float4>(material.metallicRoughnessMap, intersection.uv.x, 1.0 - intersection.uv.y);
				roughness = color.y;
				metallic = color.z;
			}
			
			bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, random, &pdf, materialColor, metallic, roughness);
		}
		else if (material.type == MaterialType::frenselSpecular)
		{
			glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
			bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, materialColor, iors);
		}
		else if (material.type == MaterialType::microfacet)
		{
			bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, material.roughness);
		}
		else//diffuse
		{
			float4 color = { 0,0,0,1 };
			if (material.baseColorMap != 0)
			{
				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
				materialColor.x = color.x;
				materialColor.y = color.y;
				materialColor.z = color.z;
			}
			if (color.w <= ALPHA_CUTOFF)
			{
				bxdf = pathSegments[idx].remainingBounces == 0 ? glm::vec3(0, 0, 0) : glm::vec3(1, 1, 1);
				wi = -wo;
				pdf = util_math_tangent_space_abscos(wi);
			}
			else
			{
				bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor);
			}
			
		}
		if (pdf > 0)
		{
			pathSegments[idx].color *= bxdf * util_math_tangent_space_abscos(wi) / pdf;
			glm::vec3 newDir = glm::normalize(TBN * wi);
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			rayValid[idx] = 1;
		}
		else
		{
			rayValid[idx] = 0;
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
		if (!util_math_is_nan(iterationPath.color))
			image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct mat_comp {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
		return a.type < b.type;
	}
};

int compact_rays(int* rayValid,int* rayIndex,int numRays, bool sortByMat=false)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths1(dev_paths1), dev_thrust_paths2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections1(dev_intersections1), dev_thrust_intersections2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(dev_thrust_paths1, dev_thrust_paths1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_paths2);
	thrust::scatter_if(dev_thrust_intersections1, dev_thrust_intersections1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersections2);
	if (sortByMat)
	{
		mat_comp cmp;
		thrust::sort_by_key(dev_thrust_intersections2, dev_thrust_intersections2 + nextNumRays, dev_thrust_paths2, cmp);
	}
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return nextNumRays;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, MAX_ITER, dev_paths1);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = dev_path_end - dev_paths1;
	int* rayValid, * rayIndex;
	
	int numRays = num_paths;
	cudaMalloc((void**)&rayValid, sizeof(int) * pixelcount);
	cudaMalloc((void**)&rayIndex, sizeof(int) * pixelcount);
	
	cudaDeviceSynchronize();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (numRays && depth < MAX_ITER) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
#if USE_BVH
#if MTBVH
		compute_intersection_bvh_stackless_mtbvh << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, numRays
			, dev_paths1
			, dev_materials
			, dev_objs
			, hst_scene->objects.size()
			, dev_triangles
			, dev_vertices
			, dev_uvs
			, dev_normals
			, dev_tangents
			, dev_fsigns
			, hst_scene->skyboxTextureObj
			, dev_primitives
			, dev_mtbvhArray
			, hst_scene->bvhTreeSize
			, dev_intersections1
			, rayValid
			, dev_image
			);
#else
		compute_intersection_bvh_stackless << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, numRays
			, dev_paths1
			, dev_materials
			, dev_objs
			, hst_scene->objects.size()
			, dev_triangles
			, dev_vertices
			, dev_uvs
			, dev_normals
			, dev_tangents
			, hst_scene->skyboxTextureObj
			, dev_primitives
			, dev_bvhArray
			, hst_scene->bvhArray.size()
			, dev_intersections1
			, rayValid
			, dev_image
			);
#endif
#else
		compute_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, numRays
			, dev_paths1
			, dev_objs
			, hst_scene->objects.size()
			, dev_triangles
			, dev_vertices
			, dev_uvs
			, dev_normals
			, hst_scene->skyboxTextureObj
			, dev_intersections1
			, rayValid
			, dev_image
			);
#endif
		cudaDeviceSynchronize();
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

#if SORT_BY_MATERIAL_TYPE
		numRays = compact_rays(rayValid, rayIndex, numRays, true);
#else
		numRays = compact_rays(rayValid, rayIndex, numRays);
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
		scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_materials,
			rayValid,
			dev_image
			);

		numRays = compact_rays(rayValid, rayIndex, numRays);

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	if (numRays)
	{
		// Assemble this iteration and apply it to the image
		dim3 numBlocksPixels = (numRays + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> > (numRays, dev_image, dev_paths1);
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	cudaFree(rayValid);
	cudaFree(rayIndex);

	checkCUDAError("pathtrace");
}
