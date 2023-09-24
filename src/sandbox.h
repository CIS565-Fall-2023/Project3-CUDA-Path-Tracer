#pragma once

#include "common.h"

class CudaPathTracer;
class Scene;
class RenderState;
class CameraController;
class GPUScene;

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
	uPtr< CameraController> m_CameraController;
	uPtr<GPUScene> m_GPUScene;

	std::string m_StartTimeString;
};