#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "triangle.h"
#include "environmentMap.h"

class Scene;
class BVH;

class GPUScene
{
public:
	CPU_ONLY GPUScene()
		:dev_triangles(nullptr),
		dev_materials(nullptr),
		dev_vertices(nullptr),
		dev_normals(nullptr),
		dev_uvs(nullptr),
		dev_BVH(nullptr),
		shape_count(0),
		material_count(0),
		light_count(0)
	{}

	GPU_ONLY Intersection SceneIntersection(const Ray& ray, int thread_id) const
	{
		//return NaiveIntersection(ray);
		return BVHIntersection_Naive(ray);
		//return BVHIntersection_Shared(ray, thread_id);
	}

	CPU_ONLY void FreeDataOnCuda();
	CPU_ONLY void Load(const Scene& scene, const BVH& bvh);

protected:
	CPU_GPU Intersection NaiveIntersection(const Ray& ray) const 
	{
		Intersection min_intersection;
		Intersection temp_intersection;
		min_intersection.t = 1000000.f;
		min_intersection.shapeId = -1;

		for (int i = 0; i < shape_count; ++i)
		{
			const glm::ivec3 idx = dev_triangles[i].v_id;
			const int material_id = dev_triangles[i].material;
			glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
			
			if (Triangle::Intersection(ray, vertices, temp_intersection) &&
				temp_intersection.t < min_intersection.t)
			{
				min_intersection = temp_intersection;
				min_intersection.shapeId = i;
				min_intersection.materialId = material_id;
			}
		}

		return min_intersection;
	}
	CPU_GPU Intersection BVHIntersection_Naive(const Ray& ray) const
	{
		const glm::vec3 inv_dir(glm::vec3(1.f) / ray.direction);
		const bool dir_neg[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

		Intersection min_intersection;
		Intersection temp_intersection;
		min_intersection.t = 10000.f;
		min_intersection.shapeId = -1;

		AABB current = dev_BVH[0];
		int stack[16];
		int next = 0;

		while (true)
		{
			float k = 20000.f;
			if (current.Intersection(ray, inv_dir, k))
			{
				if (current.m_Data.z > 0)
				{
					for (int i = 0; i < current.m_Data.y; ++i)
					{
						const glm::ivec3 idx = dev_triangles[current.m_Data.x + i].v_id;
						const int material_id = dev_triangles[current.m_Data.x + i].material;

						const glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
						if (Triangle::Intersection(ray, vertices, temp_intersection) &&
							temp_intersection.t < min_intersection.t)
						{
							min_intersection = temp_intersection;
							min_intersection.shapeId = current.m_Data.x + i;
							min_intersection.materialId = material_id;
						}
					}
					if (next == 0) break;
					current = dev_BVH[stack[--next]];
				}
				else
				{
					stack[next++] = (dir_neg[current.m_Data.x] ? (current.m_Data.y + 1) : current.m_Data.y);
					current = dev_BVH[(dir_neg[current.m_Data.x] ? current.m_Data.y : (current.m_Data.y + 1))];
				}
			}
			else
			{
				if (next == 0) break;
				current = dev_BVH[stack[--next]];
			}
		}

		return min_intersection;
	}
	GPU_ONLY Intersection BVHIntersection_Shared(const Ray& ray, int thread_id) const
	{
		const glm::vec3 inv_dir(glm::vec3(1.f) / ray.direction);
		const bool dir_neg[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

		Intersection min_intersection;
		Intersection temp_intersection;
		min_intersection.t = 10000.f;
		min_intersection.shapeId = -1;

		__shared__ int stack[16][128];
		AABB current = dev_BVH[0];
		int next = 0;
		
		while (true)
		{
			float k = 20000.f;
			if(current.Intersection(ray, inv_dir, k))
			{
				if (current.m_Data.z > 0)
				{
					for (int i = 0; i < current.m_Data.y; ++i)
					{
						const glm::ivec3 idx = dev_triangles[current.m_Data.x + i].v_id;
						const int material_id = dev_triangles[current.m_Data.x + i].material;

						const glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
						if (Triangle::Intersection(ray, vertices, temp_intersection) &&
							temp_intersection.t < min_intersection.t)
						{
							min_intersection = temp_intersection;
							min_intersection.shapeId = current.m_Data.x + i;
							min_intersection.materialId = material_id;
						}
					}
					if (next == 0) break;
					current = dev_BVH[stack[--next][thread_id]];
				}
				else
				{
					stack[next++][thread_id] = (dir_neg[current.m_Data.x] ? (current.m_Data.y + 1) : current.m_Data.y);
					current = dev_BVH[(dir_neg[current.m_Data.x] ?  current.m_Data.y : (current.m_Data.y + 1))];
				}
			}
			else
			{
				if (next == 0) break;
				current = dev_BVH[stack[--next][thread_id]];
			}
		}

		return min_intersection;
	}

public:
	TriangleIdx* dev_triangles;
	glm::vec3* dev_vertices;
	glm::vec3* dev_normals;
	glm::vec2* dev_uvs;
	Material* dev_materials;

	AABB* dev_BVH;

	EnvironmentMap env_map;

	size_t shape_count;
	size_t bvh_count;
	size_t material_count;
	size_t light_count;
};
