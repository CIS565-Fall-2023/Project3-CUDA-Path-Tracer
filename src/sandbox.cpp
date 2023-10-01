#include "sandbox.h"

#include <glm/gtx/transform.hpp>

#include "pathtrace.h"
#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"
#include "gpuScene.h"
#include "bvh.h"
#include "cameraController.h"

#include "cudaUtilities.h"

#include "ImGui/imgui.h"

#define JOIN(a, b) a##b
#define JOIN2(a, b) JOIN(a, b)
#define STR(a) #a
#define STR2(a) STR(a)

std::string resources_path = STR2(JOIN2(PROJ_BASE_PATH, /resources));

std::string currentTimeString() 
{
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

void AddCornellBox_Triangles(std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles)
{
	//int v_start_id;
	//
	//// Add light
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 0);
	//ApplyTransform(v_start_id, vertices, glm::vec3(0, 7.45, 0), glm::vec3(90, 0, 0), glm::vec3(1.5f, 1.5f, 1));
	//
	//// Add floor
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	//ApplyTransform(v_start_id, vertices, glm::vec3(0, -2.5, 0), glm::vec3(-90, 0, 0), glm::vec3(5, 5, 1));
	//
	//// Add red wall
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 2);
	//ApplyTransform(v_start_id, vertices, glm::vec3(5, 2.5, 0), glm::vec3(0, -90, 0), glm::vec3(5, 5, 1));
	//
	//// Add green wall
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 3);
	//ApplyTransform(v_start_id, vertices, glm::vec3(-5, 2.5, 0), glm::vec3(0, 90, 0), glm::vec3(5, 5, 1));
	//
	//// Add back wall
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	//ApplyTransform(v_start_id, vertices, glm::vec3(0, 2.5, 5), glm::vec3(0, 180, 0), glm::vec3(5, 5, 1));
	//
	//// Add ceiling
	//v_start_id = vertices.size();
	//AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	//ApplyTransform(v_start_id, vertices, glm::vec3(0, 7.5, 0), glm::vec3(90, 0, 0), glm::vec3(5, 5, 1));
	//
	//// Add long box
	//v_start_id = vertices.size();
	//AddBox_Triangles(v_start_id, vertices, triangles, 1);
	//ApplyTransform(v_start_id, vertices, glm::vec3(2, 0, 3), glm::vec3(0, 27.5, 0), glm::vec3(1.5, 3, 1.5));
	//
	//// Add short box
	//v_start_id = vertices.size();
	//AddBox_Triangles(v_start_id, vertices, triangles, 1);
	//ApplyTransform(v_start_id, vertices, glm::vec3(-2, -1, 0.75), glm::vec3(0, -17.5, 0), glm::vec3(1.5, 1.5, 1.5));
}

SandBox::SandBox()
	:m_PathTracer(mkU<CudaPathTracer>())
{
	m_StartTimeString = currentTimeString();

	//if (argc < 2) {
	//	printf("Usage: %s SCENEFILE.txt\n", argv[0]);
	//	return 1;
	//}
	std::filesystem::path res_path(resources_path);

	// Load scene file
	m_Scene = mkU<Scene>(res_path, "scenes/cornellMesh.json");
	m_CameraController = mkU<CameraController>(m_Scene->state.camera);

	// Set up camera stuff from loaded path tracer settings
	Camera& cam = m_Scene->state.camera;
	cam.Recompute();

	m_PathTracer->Init(m_Scene.get());

	m_GPUScene = mkU<GPUScene>();
	m_GPUScene->shape_count = m_Scene->m_TriangleIdxs.size();
	
	BVH bvh;
	bvh.Create(m_Scene->m_Vertices, m_Scene->m_TriangleIdxs);
	
	m_GPUScene->bvh_count = bvh.m_AABBs.size();

	MallocArrayOnCuda<AABB>(m_GPUScene->dev_BVH, bvh.m_AABBs);
	
	MallocArrayOnCuda<glm::vec3>(m_GPUScene->dev_vertices, m_Scene->m_Vertices);
	MallocArrayOnCuda<glm::vec3>(m_GPUScene->dev_normals, m_Scene->m_Normals);
	MallocArrayOnCuda<glm::vec2>(m_GPUScene->dev_uvs, m_Scene->m_UVs);

	MallocArrayOnCuda<TriangleIdx>(m_GPUScene->dev_triangles, m_Scene->m_TriangleIdxs);
	cudaMemcpy(bvh.m_AABBs.data(), m_GPUScene->dev_BVH, bvh.m_AABBs.size() * sizeof(AABB), cudaMemcpyDeviceToHost);
	if (m_Scene->m_EnvironmentMapId >= 0)
	{
		m_GPUScene->env_map.m_Texture.m_TexObj = m_Scene->m_Textures[m_Scene->m_EnvironmentMapId].m_TexObj;
	}
	std::cout << sizeof(Material) << std::endl;
	checkCUDAError("Copy array Error");
}

SandBox::~SandBox()
{
	m_GPUScene->FreeDataOnCuda();
}

void SandBox::OnWindowResize(int w, int h)
{
}

void SandBox::OnMouseButton(int button, int action, int mods)
{
	
}

void SandBox::OnKey(int key, int scancode, int action, int mods)
{
}

void SandBox::OnMousePosition(double x, double y)
{
	if (m_CameraController->OnMouseMoved(x, y))
	{
		m_PathTracer->Reset();
	}
}

void SandBox::OnScroll(double x, double y)
{
	if (m_CameraController->OnScroll(x, y))
	{
		m_PathTracer->Reset();
	}
}

void SandBox::DrawImGui()
{
	{
		ImGui::Begin("Path Tracer Analytics"); // Create a window called "Path Tracer Analytics" and append into it.
		ImGui::Text("Traced Depth %d", m_Scene->state.traceDepth);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();
	}
}

void SandBox::Run()
{
	if (m_PathTracer->m_Iteration == m_Scene->state.iterations) 
	{
		SaveImage(m_Scene->state.imageName.c_str());
	}
	m_PathTracer->Render(*m_GPUScene, m_Scene->state.camera);
}

void SandBox::SaveImage(const char* name)
{
	float samples = m_PathTracer->m_Iteration;
	// output image file
	glm::ivec2& res = m_Scene->state.camera.resolution;

	image img(res.x, res.y);
	m_PathTracer->GetImage(img.pixels);

	std::ostringstream ss;
	ss << name << "." << m_StartTimeString << "." << samples << "samp";

	// CHECKITOUT
	img.savePNG(ss.str());
	//img.saveHDR(ss.str());  // Save a Radiance HDR file
}
