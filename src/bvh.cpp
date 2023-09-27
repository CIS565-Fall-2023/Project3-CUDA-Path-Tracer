#include "bvh.h"

#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>

std::ostream& operator<<(std::ostream& o,const glm::vec3& v)
{
	o << "[" << v.x << ", " << v.y << ", " << v.z << "]";
	return o;
}

std::ostream& operator<<(std::ostream& o, const glm::ivec4& idx)
{
	o << "[" << idx.x << ", " << idx.y << ", " << idx.z << ", " << idx.w << "]";
	return o;
}

std::ostream& operator<<(std::ostream& o, const AABB& aabb)
{
	o << aabb.m_Min << ", " << aabb.m_Max;
	return o;
}

inline AABB TriangleAABB(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
	return { glm::min(glm::min(v0, v1), v2), glm::max(glm::max(v0, v1), v2) };
}

CPU_ONLY void BVH::Create(std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& v_ids)
{
	std::queue<std::pair<int, int>> remains;
	
	std::vector<std::pair<int, AABB>> temp_aabbs;
	std::vector<glm::ivec4> triangle_idx_ordered;

	for (int i = 0; i < v_ids.size(); ++i)
	{
		const glm::ivec4& idx = v_ids[i];
		AABB aabb = TriangleAABB(vertices[idx.x],
								 vertices[idx.y],
								 vertices[idx.z]);

		temp_aabbs.push_back({ i, aabb });
	}
	remains.push({ 0, temp_aabbs.size() });

	while (!remains.empty())
	{
		auto [start_id, end_id] = remains.front();
		remains.pop();

		if (start_id == end_id)
		{
			continue;
		}

		// leaf
		if (end_id - start_id == 1)
		{
			temp_aabbs[start_id].second.m_Data = glm::ivec3(triangle_idx_ordered.size(), 1, 1);
			m_AABBs.push_back(temp_aabbs[start_id].second);
			triangle_idx_ordered.emplace_back(v_ids[temp_aabbs[start_id].first]);
			continue;
		}

		AABB total_aabb;
		AABB center_aabb;
		for (int i = start_id; i < end_id; ++i)
		{
			total_aabb.Merge(temp_aabbs[i].second);
			center_aabb.Merge(temp_aabbs[i].second.GetCenter());
		}
		int max_axis = center_aabb.GetMaxAxis();
		
		// if the centroid bounds have zero volume, no need split
		if (center_aabb.m_Max[max_axis] == center_aabb.m_Min[max_axis])
		{
			total_aabb.m_Data = glm::ivec3(triangle_idx_ordered.size(), end_id - start_id, 1);
			m_AABBs.push_back(total_aabb);
			for (int i = start_id; i < end_id; ++i)
			{
				triangle_idx_ordered.emplace_back(v_ids[temp_aabbs[i].first]);
			}
			continue;
		}
		total_aabb.m_Data = glm::ivec3(max_axis, m_AABBs.size() + remains.size() + 1, 0);
		m_AABBs.push_back(total_aabb);

		// middle split
		float center = 0.5f * (center_aabb.m_Max[max_axis] + center_aabb.m_Min[max_axis]);;

		auto partition_func = [max_axis, center](const std::pair<int, AABB>& pair) {
			return pair.second.GetCenter()[max_axis] < center;
		};

		auto* midPtr = std::partition(&temp_aabbs[start_id], &temp_aabbs[end_id - 1] + 1, partition_func);

		int middle_id = midPtr - &temp_aabbs[0];
		if (middle_id == start_id || middle_id == end_id)
		{
			middle_id = (start_id + end_id) / 2;
			auto nth_function = [max_axis](const std::pair<int, AABB>& pa, const std::pair<int, AABB>& pb) { 
				return pa.second.GetCenter()[max_axis] < pb.second.GetCenter()[max_axis]; 
			};
			std::nth_element(&temp_aabbs[start_id], &temp_aabbs[middle_id], &temp_aabbs[end_id - 1] + 1, nth_function);
		}
		
		remains.push({ start_id, middle_id });
		remains.push({ middle_id, end_id});
	}
	v_ids.swap(triangle_idx_ordered);

	std::cout << "BVH finish" << std::endl;

	AABB current = m_AABBs[0];
	
	// stack
	int next_arr[50];
	int next = 0;
	
	//while (true)
	//{
	//	if (current.m_Data.z > 0)
	//	{
	//		for (int i = 0; i < current.m_Data.y; ++i)
	//		{
	//			const glm::ivec4& idx = v_ids[current.m_Data.x + i];
	//			std::cout << idx << "; AABB: " << current;
	//		}
	//		std::cout << std::endl;
	//		if (next == 0) break;
	//		current = m_AABBs[next_arr[--next]];
	//	}
	//	else
	//	{
	//		std::cout << "AABB: " << current << std::endl;
	//		next_arr[next++] = current.m_Data.y + 1;
	//		current = m_AABBs[current.m_Data.y];
	//	}
	//}
}
