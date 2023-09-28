#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include "sceneStructs.h"
struct BoundingBox {
	glm::vec3 minBound;
	glm::vec3 maxBound;
	BoundingBox(glm::vec3 _minB, glm::vec3 _maxB)
		:minBound(_minB),maxBound(_maxB)
	{};
	BoundingBox()
		:minBound(glm::vec3(FLT_MAX)),maxBound(glm::vec3(FLT_MIN))
	{};
	static BoundingBox unionBound(const BoundingBox& b1, const BoundingBox& b2);
	static BoundingBox unionBound(const BoundingBox& b, const glm::vec3& pt);
};

struct BVHPrimitiveInfo {
	int idx; // the index of hitabble in the overall primitive array
	BoundingBox boundingBox; // bounding box
	glm::vec3 center; // center of the bounding box
	BVHPrimitiveInfo(int idx, const BoundingBox& b)
		:idx(idx),boundingBox(b),center(b.minBound*0.5f + b.maxBound*0.5f)
	{}
};

enum PARTITION_AXIS{X,Y,Z};

struct BVHNode {
	BoundingBox boundingBox;
	BVHNode* children[2];
	PARTITION_AXIS axis;
	int primIdxBegin, primNum;
private:
	void initLeaf(int first, int n, const BoundingBox& b);
	void initInterior(PARTITION_AXIS _axis, BVHNode* c0, BVHNode* c1);
	//PARTITION_AXIS getPartitionAxis(int start, int end, const std::vector<BVHPrimitiveInfo>& primInfo);
	BVHNode(
		int start, int end,
		const std::vector<BVHPrimitiveInfo>& primInfo,
		std::vector<int>& ordered_primId,
		std::vector<std::unique_ptr<BVHNode>>& node_holder
	);
public:
	friend class BVHTree;
};

class BVHTree {
	std::vector<std::unique_ptr<BVHNode>> m_nodes;
	std::vector<BVHPrimitiveInfo> initPrimitiveInfo(const std::vector<Triangle>& trigs);
	void buildTree(const std::vector<BVHPrimitiveInfo>& primInfo);
public:
	BVHTree(const std::vector<Triangle>& trigs);
};



