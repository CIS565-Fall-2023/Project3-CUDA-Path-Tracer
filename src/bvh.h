#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "sceneStructs.h"

class BVH
{
public:
	CPU_ONLY BVH() {}

	CPU_ONLY void Create(std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& v_ids);

public:
	std::vector<AABB> m_AABBs;
};