#pragma once

#include "cudaTexture.h"

class EnvironmentMap
{
public:
	GPU_ONLY glm::vec3 Get(const glm::vec3& wiW) const
	{
		glm::vec2 uv = glm::vec2(glm::atan(wiW.z, wiW.x), glm::asin(wiW.y)) * glm::vec2(Inv2Pi, InvPi) + glm::vec2(0.5f);
		float4 value = m_Texture.Get(uv.x, uv.y);
		return { value.x, value.y, value.z };
	}
	CPU_GPU bool Valid() const { return m_Texture.m_TexObj != 0; }
public:
	CudaTexture2D  m_Texture;
};