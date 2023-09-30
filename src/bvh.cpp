#include "bvh.h"
#include <algorithm>
#include <iostream>
BoundingBox BoundingBox::unionBound(const BoundingBox& b1, const BoundingBox& b2)
{
	return BoundingBox(
		glm::vec3(
			std::min(b1.minBound.x, b2.minBound.x),
			std::min(b1.minBound.y, b2.minBound.y),
			std::min(b1.minBound.z, b2.minBound.z)
		),
		glm::vec3(
			std::max(b1.maxBound.x, b2.maxBound.x),
			std::max(b1.maxBound.y, b2.maxBound.y),
			std::max(b1.maxBound.z, b2.maxBound.z)
		)
	);
}

BoundingBox BoundingBox::unionBound(const BoundingBox& b, const glm::vec3& pt)
{
	return BoundingBox(
		glm::vec3(
			std::min(b.minBound.x, pt.x),
			std::min(b.minBound.y, pt.y),
			std::min(b.minBound.z, pt.z)
		),
		glm::vec3(
			std::max(b.maxBound.x, pt.x),
			std::max(b.maxBound.y, pt.y),
			std::max(b.maxBound.z, pt.z)
		)
	);
}

std::vector<BVHPrimitiveInfo> BVHTreeBuilder::initPrimitiveInfo(const std::vector<Triangle>& trigs)
{
	std::vector<BVHPrimitiveInfo> ans;
	int n = trigs.size();
	for (int i = 0;i < n;++i) {
		auto& trig = trigs[i];
		float minX = std::min(std::min(trig.v1.pos.x, trig.v2.pos.x), trig.v3.pos.x);
		float maxX = std::max(std::max(trig.v1.pos.x, trig.v2.pos.x), trig.v3.pos.x);
		float minY = std::min(std::min(trig.v1.pos.y, trig.v2.pos.y), trig.v3.pos.y);
		float maxY = std::max(std::max(trig.v1.pos.y, trig.v2.pos.y), trig.v3.pos.y);
		float minZ = std::min(std::min(trig.v1.pos.z, trig.v2.pos.z), trig.v3.pos.z);
		float maxZ = std::max(std::max(trig.v1.pos.z, trig.v2.pos.z), trig.v3.pos.z);
		ans.push_back(BVHPrimitiveInfo(i, BoundingBox(glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ))));
	}
	return ans;
}

int BVHTreeBuilder::recursiveLBuildTree(int start, int end, std::vector<BVHPrimitiveInfo>& primInfo)
{
	if (start == end)return -1;
	BoundingBox nodeBBox;
	for (int i = start; i < end; ++i) {
		nodeBBox = BoundingBox::unionBound(nodeBBox, primInfo[i].boundingBox);
	}
	int primNum = end - start;
	if (primNum < 5) {
		//leaf node
		m_lnodes.push_back(BVHNode(nodeBBox, LEAF_NODE, primNum, start));
		return m_lnodes.size()-1;
	}
	else {
		float maxAxisDiff = 0.f;
		PARTITION_AXIS axis = calPartitionAxis(start, end, primInfo, maxAxisDiff);
		if (maxAxisDiff < 0.01f) {
			//leaf node
			m_lnodes.push_back(BVHNode(nodeBBox, LEAF_NODE, primNum, start));
			return m_lnodes.size() - 1;
		}
		else {
			int mid = (start + end) / 2;
			std::nth_element(&primInfo[start]
				, &primInfo[mid]
				, &primInfo[end - 1] + 1,
				[axis](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
					switch (axis)
					{
					case X:
						return a.center[axis] < b.center.x;
					case Y:
						return a.center.y < b.center.y;
					case Z:
						return a.center.z < b.center.z;
					}
					return a.center.x < b.center.x;
				});
			m_lnodes.push_back(BVHNode(nodeBBox, INTERIOR_NODE, 0, -1,axis));
			int posId = m_lnodes.size() - 1;
			recursiveLBuildTree(start, mid, primInfo);
			int secondChildId = recursiveLBuildTree(mid, end, primInfo);
			if (secondChildId != -1) {
				m_lnodes[posId].secondChildOffset = secondChildId - posId;
			}
			return posId;
		}	
	}
	return -1;
}

PARTITION_AXIS BVHTreeBuilder::calPartitionAxis(int start, int end, const std::vector<BVHPrimitiveInfo>& primInfo, float& out_maxAxisDiff)
{
	BoundingBox centerBox;
	PARTITION_AXIS axis;
	for (int i = start; i < end; ++i) {
		centerBox = BoundingBox::unionBound(centerBox, primInfo[i].center);
	}
	glm::vec3 dimDiff = centerBox.maxBound - centerBox.minBound;
	if (dimDiff.x > dimDiff.y && dimDiff.x > dimDiff.z) {
		axis = X;
		out_maxAxisDiff = dimDiff.x;
	}
	else if (dimDiff.y > dimDiff.z) {
		axis = Y;
		out_maxAxisDiff = dimDiff.y;
	}
	else {
		axis = Z;
		out_maxAxisDiff = dimDiff.z;
	}
	return axis;
}

std::vector<Triangle> BVHTreeBuilder::rearrangeBasedOnPrimtiveInfo(const std::vector<BVHPrimitiveInfo>& primInfo, const std::vector<Triangle>& trigs)
{
	int n = trigs.size();
	std::vector<Triangle> ans(n);
	for (int i = 0;i < n;++i) {
		ans[i] = trigs[primInfo[i].idx];
	}
	return ans;
}

void BVHTreeBuilder::displayBVHTree(const std::vector<BVHNode>& m_lnodes, int depth, int curId)
{
	std::string space(depth*2, ' ');
	BVHNode node = m_lnodes[curId];
	if (node.primNum == 0) {
		std::cout << space << "interior: " << std::endl;
		std::cout << space << "[ " << node.boundingBox.maxBound.x << ", " << node.boundingBox.maxBound.y << ", " << node.boundingBox.maxBound.z << "]" << std::endl;
		std::cout << space << "[ " << node.boundingBox.minBound.x << ", " << node.boundingBox.minBound.y << ", " << node.boundingBox.minBound.z << "]" << std::endl;
		displayBVHTree(m_lnodes, depth + 1, curId + 1);
		if (node.secondChildOffset != -1)
			displayBVHTree(m_lnodes, depth + 1, curId + node.secondChildOffset);
	}
	else {
		std::cout << space << "leaf: " << node.firstPrimId << "->" << node.primNum << "triangles" << std::endl;
	}
}

std::vector<BVHNode> BVHTreeBuilder::buildBVHTree(std::vector<Triangle>& trigs)
{
	auto info = initPrimitiveInfo(trigs);
	if (!m_lnodes.empty())m_lnodes.clear();
	recursiveLBuildTree(0, trigs.size(), info);
	trigs = rearrangeBasedOnPrimtiveInfo(info, trigs);
	//displayBVHTree(m_lnodes, 0, 0);
	return m_lnodes;
}

BVHTreeBuilder::BVHTreeBuilder()
{}

BVHNode::BVHNode(const BoundingBox& _boundingBox, NODE_TYPE type, int _primNum, int unionVal, PARTITION_AXIS _axis)
	:boundingBox(_boundingBox), primNum(_primNum), axis(_axis)
{
	switch (type)
	{
	case LEAF_NODE:
		firstPrimId = unionVal;
		break;
	case INTERIOR_NODE:
		secondChildOffset = unionVal;
		break;
	default:
		std::cout << "Invalid node type" << std::endl;
		break;
	}
}
