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
	CPU_ONLY void Reset() { m_Iteration = 0; }
	CPU_ONLY void GetImage(uchar4* host_image);
	CPU_ONLY void RegisterPBO(unsigned int pbo);
	CPU_ONLY void UnRegisterPBO() 
	{
		cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0);
		cudaGraphicsUnregisterResource(cuda_pbo_dest_resource);
	}

	CPU_ONLY void Render(uchar4* pbo, int frame, int iter);
public:
	int m_Iteration;
	
	glm::vec3* dev_hdr_img = nullptr; // device image that map color channels to [0, inf]
	uchar4* dev_img = nullptr; // device image that map color channels to [0, 255]
	Geom* dev_geoms = nullptr;
	Material* dev_materials = nullptr;
	PathSegment* dev_paths = nullptr;
	PathSegment* dev_terminated_paths = nullptr;
	ShadeableIntersection* dev_intersections = nullptr;
	struct cudaGraphicsResource* cuda_pbo_dest_resource;

	thrust::device_ptr<PathSegment> thrust_dev_paths;
	thrust::device_ptr<PathSegment> thrust_dev_terminated_paths;

	glm::ivec2 resolution;
};