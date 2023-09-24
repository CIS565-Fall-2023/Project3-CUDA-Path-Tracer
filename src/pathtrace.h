#pragma once

#include <vector>
#include "scene.h"
#include "cuda_data.h"

class PathTracer {
private:
	Scene* hst_scene;
	GuiDataContainer* m_guiData;
	CudaMemory<glm::vec3> dev_img;
	CudaMemory<Geom> dev_geoms;
	CudaMemory<Material> dev_mat;
	CudaMemory<PathSegment> dev_path;
	CudaMemory<ShadeableIntersection> dev_intersect;
	PathTracer(const PathTracer& _pathtracer);
	PathTracer& operator=(const PathTracer& _pathtracer);
public:
	PathTracer():hst_scene(nullptr),m_guiData(nullptr){};
	//~PathTracer();
	void initDataContainer(GuiDataContainer* guiData);
	void pathtraceInit(Scene* scene);
	void pathtrace(uchar4* pbo, int frame, int iteration);
};