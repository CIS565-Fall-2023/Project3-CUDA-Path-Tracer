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

//struct BVHNode {
//	BoundingBox boundingBox;
//	BVHNode* children[2];
//	PARTITION_AXIS axis;
//	int primIdxBegin;
//	int primNum;
//	BVHNode(const BoundingBox& _boundingBox
//		, int _primIdxBegin
//		, int _primNum
//		, BVHNode* _child0 = nullptr
//		, BVHNode* _child1 = nullptr
//		, PARTITION_AXIS _axis = X
//	) :boundingBox(_boundingBox), axis(_axis), primIdxBegin(_primIdxBegin), primNum(_primNum)
//	{
//		children[0] = _child0; children[1] = _child1;
//	}
//};

enum NODE_TYPE{LEAF_NODE,INTERIOR_NODE};

struct LinearBVHNode {
	BoundingBox boundingBox;
	union
	{
		int firstPrimId;
		int secondChildOffset;//first child offset is 1
	};
	int primNum;
	PARTITION_AXIS axis;
	LinearBVHNode(
		const BoundingBox& boundingBox
		, NODE_TYPE type
		, int primNum
		, int unionVal
		, PARTITION_AXIS _axis = X);
};

class BVHTreeBuilder {
	//std::vector<std::unique_ptr<BVHNode>> m_nodes;
	std::vector<LinearBVHNode> m_lnodes;
	std::vector<BVHPrimitiveInfo> initPrimitiveInfo(const std::vector<Triangle>& trigs);
	//BVHNode* buildTree(std::vector<BVHPrimitiveInfo>& primInfo);
	//BVHNode* recursiveBuildTree(int start, int end, std::vector<BVHPrimitiveInfo>& primInfo);
	int recursiveLBuildTree(int start, int end, std::vector<BVHPrimitiveInfo>& primInfo);
	PARTITION_AXIS calPartitionAxis(int start, int end, const std::vector<BVHPrimitiveInfo>& primInfo, float& out_maxAxisDiff);
	std::vector<Triangle> rearrangeBasedOnPrimtiveInfo(const std::vector<BVHPrimitiveInfo>& primInfo, const std::vector<Triangle>& trigs);
	void displayBVHTree(const std::vector<LinearBVHNode>& m_lnodes, int depth, int curId);
public:
	BVHTreeBuilder();
	std::vector<LinearBVHNode> buildBVHTree(std::vector<Triangle>& trigs);
};



