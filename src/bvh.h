#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "sceneStructs.h"

class BVH
{
public:
	CPU_ONLY BVH() {}

	CPU_ONLY void Create(std::vector<glm::vec3>& vertices, std::vector<TriangleIdx>& v_ids);

protected:
	CPU_ONLY void MiddleSplit(const int& start_id, 
								const int& end_id, 
								int& middle_id, 
								const int& max_axis,
								std::vector<std::pair<int, AABB>>& temp_aabbs);
public:
	std::vector<AABB> m_AABBs;
};

class SAH_Bucket
{
public:
	int m_BucketId = 0;
	int m_Count = 0;
	float m_Cost = 0.f;
	AABB m_AABB;

public:
	static constexpr int Positions = 12;

	static CPU_ONLY SAH_Bucket ComputeBestSplit(const std::pair<int, AABB>* start,
												const int& count, 
												const glm::vec2& center,
												const int& max_axis);
};