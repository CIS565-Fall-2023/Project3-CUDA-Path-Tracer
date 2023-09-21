#pragma once

#include <thrust/device_ptr.h>
#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);

class GPUScene;

class CudaPathTracer
{
public:
	CPU_ONLY CudaPathTracer() {}
	CPU_ONLY ~CudaPathTracer();

	CPU_ONLY void Init(Scene* scene);
	CPU_ONLY void Render(uchar4* pbo, int frame, int iter);
	CPU_ONLY void Reset() { m_Iteration = 0.f; }
	CPU_ONLY void GetImage(uchar4* host_image);
public:
	int m_Iteration;
	uchar4* device_image = nullptr; // device image that map color channels to [0, 255]
	float3* device_hdr_image = nullptr; // device image that store color in [0, inf)

	glm::vec3* dev_image = nullptr;
	Geom* dev_geoms = nullptr;
	Material* dev_materials = nullptr;
	PathSegment* dev_paths = nullptr;
	PathSegment* dev_terminated_paths = nullptr;
	ShadeableIntersection* dev_intersections = nullptr;

	thrust::device_ptr<PathSegment> thrust_dev_paths;
	thrust::device_ptr<PathSegment> thrust_dev_terminated_paths;
};