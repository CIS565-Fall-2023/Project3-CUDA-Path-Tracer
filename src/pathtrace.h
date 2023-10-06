#pragma once

#include <thrust/device_ptr.h>
#include <vector>
#include "sceneStructs.h"

class GPUScene;
class Scene;

class CudaPathTracer
{
public:
	CPU_ONLY CudaPathTracer() {}
	CPU_ONLY ~CudaPathTracer();

	CPU_ONLY void Resize(const int& w, const int& h);
	CPU_ONLY void Init(Scene* scene);
	CPU_ONLY void Reset() { m_Iteration = 0; }
	CPU_ONLY void GetImage(uchar4* host_image);
	CPU_ONLY void RegisterPBO(unsigned int pbo);
	CPU_ONLY void UnRegisterPBO() 
	{
		cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0);
		cudaGraphicsUnregisterResource(cuda_pbo_dest_resource);
	}

	CPU_ONLY void Render(GPUScene& scene, 
						const Camera& camera,
						const UniformMaterialData& data);
public:
	int m_Iteration;
	
	glm::vec3* dev_hdr_img = nullptr; // device image that map color channels to [0, inf]
	uchar4* dev_img = nullptr; // device image that map color channels to [0, 255]

	PathSegment* dev_paths = nullptr;
	PathSegment* dev_end_paths = nullptr;
	ShadeableIntersection* dev_intersections = nullptr;
	struct cudaGraphicsResource* cuda_pbo_dest_resource = nullptr;

	thrust::device_ptr<PathSegment> thrust_dev_paths_begin;
	thrust::device_ptr<PathSegment> thrust_dev_end_paths_bgein;

	glm::ivec2 resolution;
};