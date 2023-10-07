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

enum NODE_TYPE{LEAF_NODE,INTERIOR_NODE};

struct BVHNode {
	BoundingBox boundingBox;
	union
	{
		int firstPrimId;
		int secondChildOffset;//first child offset is 1
	};
	int primNum;
	PARTITION_AXIS axis;
	BVHNode(
		const BoundingBox& boundingBox
		, NODE_TYPE type
		, int primNum
		, int unionVal
		, PARTITION_AXIS _axis = X);
};

class BVHTreeBuilder {
	std::vector<BVHNode> m_lnodes;
	std::vector<BVHPrimitiveInfo> initPrimitiveInfo(const std::vector<Triangle>& trigs);
	int recursiveLBuildTree(int start, int end, std::vector<BVHPrimitiveInfo>& primInfo);
	PARTITION_AXIS calPartitionAxis(int start, int end, const std::vector<BVHPrimitiveInfo>& primInfo, float& out_maxAxisDiff);
	std::vector<Triangle> rearrangeBasedOnPrimtiveInfo(const std::vector<BVHPrimitiveInfo>& primInfo, const std::vector<Triangle>& trigs);
	void displayBVHTree(const std::vector<BVHNode>& m_lnodes, int depth, int curId);
public:
	BVHTreeBuilder();
	std::vector<BVHNode> buildBVHTree(std::vector<Triangle>& trigs);
};



