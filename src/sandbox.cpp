#include "sandbox.h"

#include <glm/gtx/transform.hpp>

#include "scene.h"
#include "pathtrace.h"
#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"
#include "gpuScene.h"
#include "cameraController.h"

#include "cudaUtilities.h"
#include "common.h"

#define JOIN(a, b) a##b
#define JOIN2(a, b) JOIN(a, b)
#define STR(a) #a
#define STR2(a) STR(a)

std::string scene_path = STR2(JOIN2(PROJ_BASE_PATH, /scenes));

std::string currentTimeString() 
{
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

void ComputeTransform(const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale, 
						glm::mat4& transform, glm::mat3& TransposeInvTransform)
{
	glm::mat4 T = glm::translate(glm::mat4(1.f), translate);

	glm::mat4 S = glm::scale(glm::mat4(1.f), scale);

	glm::mat4 Rx = glm::rotate(glm::mat4(1.f), glm::radians(rotate.x), { 1.f, 0.f, 0.f });
	glm::mat4 Ry = glm::rotate(glm::mat4(1.f), glm::radians(rotate.y), { 0.f, 1.f, 0.f });
	glm::mat4 Rz = glm::rotate(glm::mat4(1.f), glm::radians(rotate.z), { 0.f, 0.f, 1.f });

	transform = T * Rx * Ry * Rz * S;
	TransposeInvTransform = glm::transpose(glm::inverse(glm::mat3(transform)));
}

void AddPlane_Triangles(int start_id, std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles, int material_id)
{
	vertices.emplace_back(-1, -1, 0);
	vertices.emplace_back( 1, -1, 0);
	vertices.emplace_back( 1,  1, 0);
	vertices.emplace_back(-1,  1, 0);

	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 1, 2, material_id)); // front
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 2, 3, material_id)); // front
}

void AddBox_Triangles(int start_id, std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles, int material_id)
{
	vertices.emplace_back(1, 1, -1);
	vertices.emplace_back(1, -1, -1);
	vertices.emplace_back(-1, -1, -1);
	vertices.emplace_back(-1, 1, -1);

	vertices.emplace_back(1, 1, 1);
	vertices.emplace_back(1, -1, 1);
	vertices.emplace_back(-1, -1, 1);
	vertices.emplace_back(-1, 1, 1);

	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 1, 2, material_id)); // front
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 2, 3, material_id)); // front
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(5, 4, 7, material_id)); // back
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(5, 7, 6, material_id)); // back
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(6, 7, 3, material_id)); // right
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(6, 3, 2, material_id)); // right
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 5, 1, material_id)); // left
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 4, 5, material_id)); // left
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(3, 7, 4, material_id)); // top
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(3, 4, 0, material_id)); // top
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(2, 1, 5, material_id)); // bottom
	triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(2, 5, 6, material_id)); // bottom
}

void ApplyTransform(int start_id, std::vector<glm::vec3>& vertices, 
					const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale)
{
	glm::mat4 transform;
	glm::mat3 inv_transpose;
	ComputeTransform(translate, rotate, scale, transform, inv_transpose);
	for (int i = start_id; i < vertices.size(); ++i)
	{
		vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
	}
}

void AddCornellBox_Triangles(std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles)
{
	int v_start_id;
	
	// Add light
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 0);
	ApplyTransform(v_start_id, vertices, glm::vec3(0, 7.45, 0), glm::vec3(90, 0, 0), glm::vec3(1.5f, 1.5f, 1));

	// Add floor
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	ApplyTransform(v_start_id, vertices, glm::vec3(0, -2.5, 0), glm::vec3(-90, 0, 0), glm::vec3(5, 5, 1));
	
	// Add red wall
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 2);
	ApplyTransform(v_start_id, vertices, glm::vec3(5, 2.5, 0), glm::vec3(0, -90, 0), glm::vec3(5, 5, 1));
	
	// Add green wall
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 3);
	ApplyTransform(v_start_id, vertices, glm::vec3(-5, 2.5, 0), glm::vec3(0, 90, 0), glm::vec3(5, 5, 1));
	
	// Add back wall
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	ApplyTransform(v_start_id, vertices, glm::vec3(0, 2.5, 5), glm::vec3(0, 180, 0), glm::vec3(5, 5, 1));
	
	// Add ceiling
	v_start_id = vertices.size();
	AddPlane_Triangles(v_start_id, vertices, triangles, 1);
	ApplyTransform(v_start_id, vertices, glm::vec3(0, 7.5, 0), glm::vec3(90, 0, 0), glm::vec3(5, 5, 1));

	// Add long box
	v_start_id = vertices.size();
	AddBox_Triangles(v_start_id, vertices, triangles, 1);
	ApplyTransform(v_start_id, vertices, glm::vec3(2, 0, 3), glm::vec3(0, 27.5, 0), glm::vec3(1.5, 3, 1.5));
	
	// Add short box
	v_start_id = vertices.size();
	AddBox_Triangles(v_start_id, vertices, triangles, 1);
	ApplyTransform(v_start_id, vertices, glm::vec3(-2, -1, 0.75), glm::vec3(0, -17.5, 0), glm::vec3(1.5, 1.5, 1.5));
}

SandBox::SandBox()
	:m_PathTracer(mkU<CudaPathTracer>())
{
	m_StartTimeString = currentTimeString();

	//if (argc < 2) {
	//	printf("Usage: %s SCENEFILE.txt\n", argv[0]);
	//	return 1;
	//}
	std::string scene_file = (scene_path + "/cornellBox.txt");
	const char* sceneFile = scene_file.c_str();

	// Load scene file
	m_Scene = mkU<Scene>(sceneFile);
	m_CameraController = mkU<CameraController>(m_Scene->state.camera);

	// Set up camera stuff from loaded path tracer settings
	Camera& cam = m_Scene->state.camera;
	cam.Recompute();

	m_PathTracer->Init(m_Scene.get());

	m_GPUScene = mkU<GPUScene>();

	std::vector<glm::vec3> vertices;
	std::vector<glm::ivec4> triangles;

	AddCornellBox_Triangles(vertices, triangles);

	MallocArrayOnCuda<glm::vec3>(m_GPUScene->dev_vertices, vertices);
	MallocArrayOnCuda<glm::ivec4>(m_GPUScene->dev_triangles, triangles);
	
	checkCUDAError("Copy array Error");

	m_GPUScene->shape_count = triangles.size();
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

void SandBox::DrawImGui()
{
}

void SandBox::Run()
{
	if (m_PathTracer->m_Iteration < m_Scene->state.iterations) 
	{
		m_PathTracer->Render(*m_GPUScene);
	}
	else 
	{
		SaveImage(m_Scene->state.imageName.c_str());
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
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
