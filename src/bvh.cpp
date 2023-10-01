#include "sceneStructs.h"
#include "bvh.h"
#include "utilities.h"
#include <vector>


BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2)
{
	BoundingBox b;
	b.pMin = glm::min(b1.pMin, b2.pMin);
	b.pMax = glm::max(b1.pMax, b2.pMax);
	return b;
}

BoundingBox Union(const BoundingBox& b1, const glm::vec3& p)
{
	BoundingBox b;
	b.pMin = glm::min(b1.pMin, p);
	b.pMax = glm::max(b1.pMax, p);
	return b;
}
//TODO: check if use projected area is better
float BoxArea(const BoundingBox& b)
{
	glm::vec3 diff = b.pMax - b.pMin;
	return 2 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
}

Primitive::Primitive(const Object& obj, int objID, int triangleOffset, const glm::ivec3* triangles, const glm::vec3* vertices)
{
	this->objID = objID;
	this->offset = triangleOffset;
	if (obj.type == SPHERE || obj.type == CUBE)
	{
		static const glm::vec3 verts[8] = { glm::vec3(0.5,0.5,0.5),glm::vec3(-0.5,0.5,0.5),glm::vec3(0.5,-0.5,0.5),glm::vec3(0.5,0.5,-0.5),glm::vec3(0.5,-0.5,-0.5),glm::vec3(-0.5,0.5,-0.5),glm::vec3(0.5,-0.5,-0.5),glm::vec3(-0.5,-0.5,-0.5) };
		for (int i = 0; i < 8; i++)
		{
			const glm::vec3& tmp = glm::vec3(obj.Transform.transform * glm::vec4(verts[i], 1.0));
			bbox = Union(bbox, tmp);
		}
	}
	else if (obj.type == TRIANGLE_MESH)
	{
		const glm::ivec3& tri = triangles[obj.triangleStart+offset];
		for (int i = 0; i < 3; i++)
		{
			const glm::vec3& vert = vertices[tri[i]];
			const glm::vec3& tmp = glm::vec3(obj.Transform.transform * glm::vec4(vert, 1.0));
			bbox = Union(bbox, tmp);
		}
	}
	bbox.pMin -= glm::vec3(BOUNDING_BOX_EXPAND);
	bbox.pMax += glm::vec3(BOUNDING_BOX_EXPAND);
	assert(bbox.pMin.x < bbox.pMax.x && bbox.pMin.y < bbox.pMax.y && bbox.pMin.z < bbox.pMax.z);
}

BVHNode* buildBVHTreeRecursiveSAH(std::vector<Primitive>& primitives, int start, int end, int* size)
{
	BVHNode* root = new BVHNode();
	(*size)++;
	BoundingBox bb;
	for (int i = start; i < end; i++)
	{
		bb = Union(bb, primitives[i].bbox);
	}
	root->bbox = bb;
	BoundingBox bCenter;
	for (int i = start; i < end; i++)
	{
		bCenter = Union(bCenter, primitives[i].bbox.center());
	}
	int numPrims = end - start;
	glm::vec3 centerDiff = bCenter.pMax - bCenter.pMin;
	int axis = centerDiff.x >= centerDiff.y && centerDiff.x >= centerDiff.z ? 0 : (centerDiff.y >= centerDiff.z ? 1 : 2);
	root->axis = axis;
	root->startPrim = start;
	root->endPrim = end;

	if(numPrims > MAX_NUM_PRIMS_IN_LEAF && centerDiff[axis] > EPSILON)
	{
		struct SAHBucket {
			int cnt = 0;
			BoundingBox bbox;
		};
		SAHBucket buckets[SAH_BUCKET_SIZE];
		for (int i = start; i < end; i++) 
		{
			int bidx = (primitives[i].bbox.center()[axis] - bCenter.pMin[axis]) / (centerDiff[axis]) * SAH_BUCKET_SIZE;
			if (bidx == SAH_BUCKET_SIZE) bidx--;
			buckets[bidx].cnt++;
			buckets[bidx].bbox = Union(buckets[bidx].bbox, primitives[i].bbox);
		}
		float splitCost[SAH_BUCKET_SIZE - 1];
		for (int i = 0; i < SAH_BUCKET_SIZE - 1; i++)
		{
			BoundingBox b0, b1;
			int cnt0 = 0, cnt1 = 0;
			for (int j = 0; j <= i; j++)
			{
				cnt0 += buckets[j].cnt;
				b0 = Union(b0, buckets[j].bbox);
			}
			for (int j = i+1; j < SAH_BUCKET_SIZE; j++)
			{
				cnt1 += buckets[j].cnt;
				b1 = Union(b1, buckets[j].bbox);
			}
			splitCost[i] = SAH_RAY_BOX_INTERSECTION_COST + (cnt0 * BoxArea(b0) + cnt1 * BoxArea(b1)) / BoxArea(root->bbox);
		}
		int minSplitCostIdx = std::min_element(splitCost, splitCost + SAH_BUCKET_SIZE - 1) - splitCost;
		float minCost = splitCost[minSplitCostIdx];
		float makeLeafCost = numPrims;
		if (minCost< makeLeafCost || numPrims>MAX_NUM_PRIMS_IN_LEAF)
		{
			auto midit = std::partition(primitives.begin() + start,
				primitives.begin() + end,
				[&](const Primitive& prim) {
					int bidx = (prim.bbox.center()[axis] - bCenter.pMin[axis]) / (centerDiff[axis]) * SAH_BUCKET_SIZE;
					if (bidx == SAH_BUCKET_SIZE) bidx = SAH_BUCKET_SIZE - 1;
					return bidx <= minSplitCostIdx;
				});
			int mid = midit - primitives.begin();
			root->left = buildBVHTreeRecursiveSAH(primitives, start, mid, size);
			root->right = buildBVHTreeRecursiveSAH(primitives, mid, end, size);
		}
	}

	return root;
}

void destroyBVHTree(BVHNode* root)
{
	if (!root) return;
	BVHNode* left = root->left;
	BVHNode* right = root->right;
	destroyBVHTree(left);
	destroyBVHTree(right);
	delete root;
}

bool checkBVHTreeFull(BVHNode* root)
{
	if (!root) return true;
	if (root->left && root->right)
		return checkBVHTreeFull(root->left) && checkBVHTreeFull(root->right);
	if (!root->left && !root->right) return true;
	return false;
}

int recursiveCompactBVHTreeForStacklessTraverse(std::vector<BVHGPUNode>& bvhArray, BVHNode* root, int parent)
{
	if (!root) return -1;
	BVHGPUNode node;
	node.axis = root->axis;
	node.bbox = root->bbox;
	node.parent = parent;
	node.startPrim = root->startPrim;
	node.endPrim = root->endPrim;

	int curr = bvhArray.size();
	bvhArray.emplace_back(node);
	int left = recursiveCompactBVHTreeForStacklessTraverse(bvhArray, root->left, curr);
	int right = recursiveCompactBVHTreeForStacklessTraverse(bvhArray, root->right, curr);
	bvhArray[curr].left = left;
	bvhArray[curr].right = right;
	return curr;
}


static int recursiveCompactBVHTreeToMTBVH(MTBVHGPUNode* MTBVHArr, BVHNode* root, int curr, int dir)
{
	if (!root) return -1;
	int axis = abs(dirs[dir]) - 1;
	int sgn = dirs[dir] > 0 ? 1 : -1;
	int next;
	MTBVHArr[curr].bbox = root->bbox;
	if (root->left && root->right)
	{
		BVHNode* nextHitNode = root->left->bbox.center()[axis] * sgn < root->right->bbox.center()[axis] * sgn ? root->left : root->right;
		MTBVHArr[curr].hitLink = curr + 1;
		next = recursiveCompactBVHTreeToMTBVH(MTBVHArr, nextHitNode, curr + 1, dir);
		next = recursiveCompactBVHTreeToMTBVH(MTBVHArr, root->left != nextHitNode ? root->left : root->right, next, dir);
		MTBVHArr[curr].missLink = next;
	}
	else//leaf node
	{
		next = curr + 1;
		MTBVHArr[curr].hitLink = next;
		MTBVHArr[curr].missLink = next;
		MTBVHArr[curr].startPrim = root->startPrim;
		MTBVHArr[curr].endPrim = root->endPrim;
	}
	return next;
}

void compactBVHTreeToMTBVH(std::vector<MTBVHGPUNode>& MTBVHGPUNode, BVHNode* root, int treeSize)
{
	MTBVHGPUNode.resize(6 * treeSize);
	for (int d = 0; d < 6; d++)
	{
		int next = recursiveCompactBVHTreeToMTBVH(&MTBVHGPUNode[d * treeSize], root, 0, d);
		assert(next == treeSize);
	}
}

