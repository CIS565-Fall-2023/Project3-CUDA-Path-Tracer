#include "cameraController.h"

#include "sceneStructs.h"
#include "application.h"
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

CameraController::CameraController(Camera& camera, 
									float pan_speed, 
									float zoom_speed, 
									float rotate_speed)
	:m_Camera(camera), 
	m_PanSpeed(pan_speed), 
	m_ZoomSpeed(zoom_speed), 
	m_RotateSpeed(rotate_speed)
{
}

bool CameraController::OnMouseMoved(double x, double y)
{
	glm::vec2 cur_pos = glm::vec2(x, y);
	glm::vec2 offset = m_MousePos - cur_pos;
	m_MousePos = cur_pos;

	if (Application::GetMouseState(GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		if (Application::GetKeyState(GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		{
			glm::vec2 rotate = offset * m_RotateSpeed;
			RotatePosition(-rotate.x, m_Camera.up);
			RotatePosition(rotate.y, m_Camera.right);
			RecomputeCamera();
			return true;
		}
		else if (Application::GetKeyState(GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		{
			glm::vec2 translate = offset * m_PanSpeed;
			TranslatePositionAlong(-translate.x, m_Camera.right);
			TranslatePositionAlong(translate.y, m_Camera.up);
			TranslateRefAlong(-translate.x, m_Camera.right);
			TranslateRefAlong(translate.y, m_Camera.up);
			RecomputeCamera();
			return true;
		}
	}
	return false;
}

void CameraController::RecomputeCamera()
{
	m_Camera.Recompute();
}

void CameraController::RotateRef(const float& degree, const glm::vec3& axis)
{
	// rotate ref point around camera position
	m_Camera.ref -= m_Camera.position;
	m_Camera.ref = glm::vec3(glm::rotate(glm::mat4(), degree, axis) * glm::vec4(m_Camera.ref, 1.f));
	m_Camera.ref += m_Camera.position;
}

void CameraController::RotatePosition(const float& degree, const glm::vec3& axis)
{
	// rotate camera position around ref point
	m_Camera.position -= m_Camera.ref;
	m_Camera.position = glm::vec3(glm::rotate(glm::mat4(), degree, axis) * glm::vec4(m_Camera.position, 1.f));
	m_Camera.position += m_Camera.ref;
}

void CameraController::TranslatePositionAlong(float amount, const glm::vec3& axis)
{
	glm::vec3 translate = amount * axis;
	m_Camera.position += translate;
}

void CameraController::TranslateRefAlong(float amount, const glm::vec3& axis)
{
	glm::vec3 translate = amount * axis;
	m_Camera.ref += translate;
}
