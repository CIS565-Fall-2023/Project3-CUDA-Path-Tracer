#pragma once

#include <glm/glm.hpp>

class Camera;

class CameraController
{
public:
	CameraController(Camera& camera, 
						float pan_speed = 0.05f, 
						float zoom_speed = 0.02f, 
						float rotate_speed = 0.02f);

	bool OnMouseMoved(double x, double y);
	void RecomputeCamera();
private:
	void RotateRef(const float& degree, const glm::vec3& axis);
	void RotatePosition(const float& degree, const glm::vec3& axis);

	void TranslatePositionAlong(float amount, const glm::vec3& axis);
	void TranslateRefAlong(float amount, const glm::vec3& axis);

public:
	float m_PanSpeed;
	float m_ZoomSpeed;
	float m_RotateSpeed;

protected:
	Camera& m_Camera;
	glm::vec2 m_MousePos;
};