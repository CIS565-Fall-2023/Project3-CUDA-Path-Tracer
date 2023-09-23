#include "sandbox.h"

#include <glm/gtx/transform.hpp>

#include "scene.h"
#include "pathtrace.h"
#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"

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

SandBox::SandBox()
{
	m_PathTracer = mkU<CudaPathTracer>();

	m_StartTimeString = currentTimeString();

	//if (argc < 2) {
	//	printf("Usage: %s SCENEFILE.txt\n", argv[0]);
	//	return 1;
	//}
	std::string scene_file = (scene_path + "/cornellBox.txt");
	const char* sceneFile = scene_file.c_str();

	// Load scene file
	m_Scene = mkU<Scene>(sceneFile);

	// Set up camera stuff from loaded path tracer settings
	Camera& cam = m_Scene->state.camera;

	int width = cam.resolution.x;
	int height = cam.resolution.y;
	
	glm::vec3 view = cam.forward;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);
	
	cameraPosition = cam.position;
	
	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.ref;
	zoom = glm::length(cam.position - ogLookAt);

	m_PathTracer->Init(m_Scene.get());
}

SandBox::~SandBox()
{
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
}

void SandBox::DrawImGui()
{
}

void SandBox::Run()
{
	if (camchanged) {
		Camera& cam = m_Scene->state.camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.forward = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.forward;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.ref;
		cam.position = cameraPosition;
		camchanged = false;

		m_PathTracer->Reset();
	}

	if (m_PathTracer->m_Iteration < m_Scene->state.iterations) {
		m_PathTracer->Render(nullptr, 0, 0);
	}
	else {
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
