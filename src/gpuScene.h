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
		shape_count(0),
		material_count(0),
		light_count(0)
	{}

	CPU_GPU Intersection SceneIntersection(const Ray& ray)
	{
		return NaiveIntersection(ray);
	}

	CPU_ONLY void FreeDataOnCuda()
	{
		SafeCudaFree(dev_triangles);
		SafeCudaFree(dev_vertices);
		SafeCudaFree(dev_normals);
		SafeCudaFree(dev_uvs);
	}

protected:
	CPU_GPU Intersection NaiveIntersection(const Ray& ray)
	{
		Intersection min_intersection;
		Intersection temp_intersection;

		min_intersection.t = 10000000000.f;
		min_intersection.shapeId = -1;

		for (int i = 0; i < shape_count; ++i)
		{
			glm::ivec4 idx = dev_triangles[i];
			glm::vec3 vertices[3]{ dev_vertices[idx.x], dev_vertices[idx.y], dev_vertices[idx.z] };
			
			if (Triangle::Intersection(ray, glm::vec3(idx), vertices, temp_intersection) &&
				temp_intersection.t < min_intersection.t)
			{
				min_intersection = temp_intersection;
				min_intersection.shapeId = i;
				min_intersection.materialId = idx.a;
			}
		}

		return min_intersection;
	}

public:
	glm::ivec4* dev_triangles;
	glm::vec3* dev_vertices;
	glm::vec3* dev_normals;
	glm::vec2* dev_uvs;

	size_t shape_count;
	size_t material_count;
	size_t light_count;
};