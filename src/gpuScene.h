#pragma once

#include <cuda_runtime.h>
#include "common.h"
#include "triangle.h"

class GPUScene
{
public:
	CPU_ONLY GPUScene()
		:dev_triangles(nullptr),
		dev_vertices(nullptr),
		dev_normals(nullptr),
		dev_uvs(nullptr),
		dev_BVH(nullptr),
		shape_count(0),
		material_count(0),
		light_count(0)
	{}

	CPU_GPU Intersection SceneIntersection(const Ray& ray)
	{
		//return NaiveIntersection(ray);
		return BVHIntersection(ray);
	}

	CPU_ONLY void FreeDataOnCuda()
	{
		SafeCudaFree(dev_triangles);
		SafeCudaFree(dev_vertices);
		SafeCudaFree(dev_normals);
		SafeCudaFree(dev_uvs);
		SafeCudaFree(dev_BVH);
	}

protected:
	CPU_GPU Intersection NaiveIntersection(const Ray& ray)
	{
		Intersection min_intersection;
		Intersection temp_intersection;
		min_intersection.t = 1000000.f;
		min_intersection.shapeId = -1;

		for (int i = 0; i < shape_count; ++i)
		{
			glm::ivec4 idx = dev_triangles[i];
			glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
			
			if (Triangle::Intersection(ray, vertices, temp_intersection) &&
				temp_intersection.t < min_intersection.t)
			{
				min_intersection = temp_intersection;
				min_intersection.shapeId = i;
				min_intersection.materialId = idx.a;
			}
		}

		return min_intersection;
	}

	CPU_GPU Intersection BVHIntersection(const Ray& ray)
	{
		const glm::vec3 inv_dir(glm::vec3(1.f) / ray.direction);
		const bool dir_neg[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

		Intersection min_intersection;
		Intersection temp_intersection;
		min_intersection.t = 10000.f;
		min_intersection.shapeId = -1;

		/*for (int i = 0; i < bvh_count; ++i)
		{
			AABB current = dev_BVH[i];
			float k = 2000.f;
			if ((current.Intersection(ray, inv_dir, k) && k < min_intersection.t))
			{
				if (current.m_Data.z > 0)
				{
					for (int i = 0; i < current.m_Data.y; ++i)
					{
						glm::ivec4 idx = dev_triangles[current.m_Data.x + i];
						glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
						if (Triangle::Intersection(ray,vertices, temp_intersection) &&
							temp_intersection.t < min_intersection.t)
						{
							min_intersection = temp_intersection;
							min_intersection.shapeId = current.m_Data.x + i;
							min_intersection.materialId = idx.a;
						}
					}
				}
			}
		}*/

		AABB current = dev_BVH[0];
		// stack
		int next_arr[64];
		int next = 0;
		
		while (true)
		{
			float k = 20000.f;
			if(current.Intersection(ray, inv_dir, k))
			{
				//current.Intersection(ray, inv_dir, k);
				//if(k < 20000.f) printf("%f!\n", k);
				if (current.m_Data.z > 0)
				{
					for (int i = 0; i < current.m_Data.y; ++i)
					{
						glm::ivec4 idx = dev_triangles[current.m_Data.x + i];
						glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
						if (Triangle::Intersection(ray, vertices, temp_intersection) &&
							temp_intersection.t < min_intersection.t)
						{
							min_intersection = temp_intersection;
							min_intersection.shapeId = current.m_Data.x + i;
							min_intersection.materialId = idx.a;
						}
					}
					if (next == 0) break;
					current = dev_BVH[next_arr[--next]];
				}
				else
				{
					next_arr[next++] = (dir_neg[current.m_Data.x] ? (current.m_Data.y + 1) : current.m_Data.y);
					current = dev_BVH[(dir_neg[current.m_Data.x] ? current.m_Data.y : (current.m_Data.y + 1))];
				}
			}
			else
			{
				if (next == 0) break;
				current = dev_BVH[next_arr[--next]];
			}
		}

		return min_intersection;
	}

public:
	glm::ivec4* dev_triangles;
	glm::vec3* dev_vertices;
	glm::vec3* dev_normals;
	glm::vec2* dev_uvs;

	AABB* dev_BVH;

	size_t shape_count;
	size_t bvh_count;
	size_t material_count;
	size_t light_count;
};