#include "gpuScene.h"
#include "scene.h"

#include "cudaUtilities.h"
#include "bvh.h"

CPU_ONLY void GPUScene::FreeDataOnCuda()
{
	SafeCudaFree(dev_triangles);
	SafeCudaFree(dev_materials);
	SafeCudaFree(dev_vertices);
	SafeCudaFree(dev_normals);
	SafeCudaFree(dev_uvs);
	SafeCudaFree(dev_BVH);
}

CPU_ONLY void GPUScene::Load(const Scene& scene, const BVH& bvh)
{
	FreeDataOnCuda();

	shape_count = scene.m_TriangleIdxs.size();
	material_count = scene.materials.size();
	bvh_count = bvh.m_AABBs.size();

	MallocArrayOnCuda<glm::vec3>(dev_vertices, scene.m_Vertices);
	MallocArrayOnCuda<glm::vec3>(dev_normals, scene.m_Normals);
	MallocArrayOnCuda<glm::vec2>(dev_uvs, scene.m_UVs);
	MallocArrayOnCuda<TriangleIdx>(dev_triangles, scene.m_TriangleIdxs);
	MallocArrayOnCuda<Material>(dev_materials, scene.materials);
	MallocArrayOnCuda<AABB>(dev_BVH, bvh.m_AABBs);
	
	checkCUDAError("Load GPU scene Error!");

	if (scene.m_EnvironmentMapId >= 0)
	{
		env_map.m_Texture.m_TexObj = scene.m_Textures[scene.m_EnvironmentMapId].m_TexObj;
	}
}
