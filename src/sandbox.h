#pragma once

#include "common.h"

class CudaPathTracer;
class Scene;
class RenderState;

class SandBox
{
public:
	SandBox();
	~SandBox();

	void OnWindowResize(int w, int h);
	void OnMouseButton(int button, int action, int mods);
	void OnKey(int key, int scancode, int action, int mods);
	void OnMousePosition(double x, double y);

	void DrawImGui();
	void Run();

protected:
	void SaveImage(const char* name);

public:
	uPtr<Scene> m_Scene;
	uPtr<CudaPathTracer> m_PathTracer;

	std::string m_StartTimeString;

	float zoom, theta, phi;
	glm::vec3 cameraPosition;
	glm::vec3 ogLookAt; // for recentering the camera
	bool camchanged = true;
};