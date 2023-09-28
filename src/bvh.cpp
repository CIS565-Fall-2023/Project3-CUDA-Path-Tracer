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

CPU_ONLY void BVH::MiddleSplit(const int& start_id,
								const int& end_id,
								int& middle_id,
								const int& max_axis,
								std::vector<std::pair<int, AABB>>& temp_aabbs)
{
	middle_id = (start_id + end_id) / 2;
	std::nth_element(&temp_aabbs[start_id], &temp_aabbs[middle_id], &temp_aabbs[end_id - 1] + 1, 
		[max_axis](const std::pair<int, AABB>& pa, const std::pair<int, AABB>& pb) {
			return pa.second.GetCenter()[max_axis] < pb.second.GetCenter()[max_axis];
	});
}

CPU_ONLY void BVH::Create(std::vector<glm::vec3>& vertices, std::vector<TriangleIdx>& v_ids)
{
	std::queue<std::pair<int, int>> remains;
	
	std::vector<std::pair<int, AABB>> temp_aabbs;
	std::vector<TriangleIdx> triangle_idx_ordered;

	for (int i = 0; i < v_ids.size(); ++i)
	{
		const glm::ivec3& idx = v_ids[i].v_id;
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
		int triangle_count = end_id - start_id;
		assert(triangle_count >= 0);

		if (triangle_count == 0)
		{
			continue;
		}

		// leaf
		if (triangle_count == 1)
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
			total_aabb.m_Data = glm::ivec3(triangle_idx_ordered.size(), triangle_count, 1);
			m_AABBs.push_back(total_aabb);
			for (int i = start_id; i < end_id; ++i)
			{
				triangle_idx_ordered.emplace_back(v_ids[temp_aabbs[i].first]);
			}
			continue;
		}

#if SAH_BVH
		int middle_id;
		if (triangle_count <= 4)
		{
			MiddleSplit(start_id, end_id, middle_id, max_axis, temp_aabbs);
		}
		else
		{
			// compute cost for each splitting position 
			// to get the best splitting position		
			glm::vec2 center(center_aabb.m_Min[max_axis], center_aabb.m_Max[max_axis]);
			SAH_Bucket min_bucket = SAH_Bucket::ComputeBestSplit(&temp_aabbs[start_id], triangle_count, center, max_axis);
			min_bucket.m_Cost = 0.125 * min_bucket.m_Cost / total_aabb.GetCost();
			const int& best_pos = min_bucket.m_BucketId;
			if (triangle_count <= 255 && triangle_count <= min_bucket.m_Cost)
			{
				// create a leaf node
				total_aabb.m_Data = glm::ivec3(triangle_idx_ordered.size(), triangle_count, 1);
				m_AABBs.push_back(total_aabb);

				for (int i = start_id; i < end_id; ++i)
				{
					triangle_idx_ordered.emplace_back(v_ids[temp_aabbs[i].first]);
				}
				continue;
			}

			auto partition_func = [max_axis, center, best_pos](const std::pair<int, AABB>& pair) {
				int pos = SAH_Bucket::Positions * (pair.second.GetCenter()[max_axis] - center.x) / (center.y - center.x);
				return pos <= best_pos;
			};

			// partition
			auto* midPtr = std::partition(&temp_aabbs[start_id], &temp_aabbs[end_id - 1] + 1, partition_func);
			middle_id = midPtr - temp_aabbs.data();
		}
#else
		// middle split
		float center = 0.5f * (center_aabb.m_Max[max_axis] + center_aabb.m_Min[max_axis]);;

		auto partition_func = [max_axis, center](const std::pair<int, AABB>& pair) {
			return pair.second.GetCenter()[max_axis] < center;
		};

		auto* midPtr = std::partition(&temp_aabbs[start_id], &temp_aabbs[end_id - 1] + 1, partition_func);

		int middle_id = midPtr - temp_aabbs.data();
		if (middle_id == start_id || middle_id == end_id)
		{
			MiddleSplit(start_id, end_id, middle_id, max_axis, temp_aabbs);
		}
#endif
		total_aabb.m_Data = glm::ivec3(max_axis, m_AABBs.size() + remains.size() + 1, 0);
		m_AABBs.push_back(total_aabb);

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

CPU_ONLY SAH_Bucket SAH_Bucket::ComputeBestSplit(const std::pair<int, AABB>* start,
												 const int& count,
												 const glm::vec2& center,
												 const int& max_axis)
{
	SAH_Bucket buckets[SAH_Bucket::Positions];
	for (int i = 0; i < count; ++i)
	{
		const AABB& aabb = (start + i)->second;

		int pos = SAH_Bucket::Positions * (aabb.GetCenter()[max_axis] - center.x) / (center.y - center.x);
		pos = std::clamp(pos, 0, SAH_Bucket::Positions - 1);
		++buckets[pos].m_Count;
		buckets[pos].m_AABB.Merge(aabb);
	}

	const SAH_Bucket* min_bucket = &buckets[0];

	for (int i = 0; i < SAH_Bucket::Positions; ++i)
	{
		AABB left, right;
		int l_count = 0, r_count = 0;

		for (int j = 0; j <= i; ++j)
		{
			left.Merge(buckets[j].m_AABB);
			l_count += buckets[j].m_Count;
		}

		for (int j = i + 1; j < SAH_Bucket::Positions; ++j)
		{
			right.Merge(buckets[j].m_AABB);
			r_count += buckets[j].m_Count;
		}

		buckets[i].m_BucketId = i;
		buckets[i].m_Cost = l_count * left.GetCost() + r_count * right.GetCost();
		if (buckets[i].m_Cost < min_bucket->m_Cost)
		{
			min_bucket = &buckets[i];
		}
	}

	return *min_bucket;
}