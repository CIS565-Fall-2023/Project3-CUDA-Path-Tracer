#include "bvh.h"
#include <algorithm>
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

void BVHNode::initLeaf(int first, int n, const BoundingBox& b)
{
	primIdxBegin = first;
	primNum = n;
	boundingBox = b;
	children[0] = children[1] = nullptr;
}

void BVHNode::initInterior(PARTITION_AXIS _axis, BVHNode* c0, BVHNode* c1)
{
	children[0] = c0;
	children[1] = c1;
	boundingBox = BoundingBox::unionBound(c0->boundingBox, c1->boundingBox);
	axis = _axis;
	primNum = 0;
}

BVHNode::BVHNode(int start, int end, const std::vector<BVHPrimitiveInfo>& primInfo, std::vector<int>& ordered_primId)
	:primNum(end - start)
{
	//for (int i = start; i < end; ++i) {
	//	boundingBox = BoundingBox::unionBound(boundingBox, primInfo[i].boundingBox);
	//}
	//if (primNum == 1) {
	//	int startId = ordered_primId.size();
	//	for (int i = start;i < end;++i) {
	//		ordered_primId.push_back(primInfo[i].idx);
	//	}
	//	initLeaf(startId, primNum, boundingBox);
	//	return;
	//}
	//else {
	//	BoundingBox centerBox;
	//	for (int i = start; i < end; ++i) {
	//		centerBox = BoundingBox::unionBound(centerBox, primInfo[i].center);
	//	}
	//	PARTITION_AXIS axis;
	//	glm::vec3 dimDiff = centerBox.maxBound - centerBox.minBound;
	//	float axisMaxDiff = 0;
	//	if (dimDiff.x > dimDiff.y && dimDiff.x > dimDiff.z) {
	//		axis = X;
	//		axisMaxDiff = dimDiff.x;
	//	}
	//	else if (dimDiff.y > dimDiff.z) {
	//		axis = Y;
	//		axisMaxDiff = dimDiff.y;
	//	}
	//	else {
	//		axis = Z;
	//		axisMaxDiff = dimDiff.z;
	//	}
	//	if (axisMaxDiff < 0.01) {
	//		int startId = ordered_primId.size();
	//		for (int i = start;i < end;++i) {
	//			ordered_primId.push_back(primInfo[i].idx);
	//		}
	//		initLeaf(startId, primNum, boundingBox);
	//		return;
	//	}
	//	else {
	//		int mid = (start + end) / 2;
	//		std::nth_element(&primInfo[start]
	//			, &primInfo[mid]
	//			, &primInfo[end - 1] + 1,
	//			[axis](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
	//				switch (axis)
	//				{
	//				case X:
	//					return a.center.x < b.center.x;
	//				case Y:
	//					return a.center.y < b.center.y;
	//				case Z:
	//					return a.center.z < b.center.z;
	//				}
	//				return a.center.x < b.center.x;
	//			});
	//		children[0] = new BVHNode(start, mid, primInfo, ordered_primId);
	//		children[1] = new BVHNode(mid, end, primInfo, ordered_primId);
	//		initInterior(axis, children[0], children[1]);
	//	}
	//}
}

BVHNode::~BVHNode()
{
	delete children[0];
	delete children[1];
}
