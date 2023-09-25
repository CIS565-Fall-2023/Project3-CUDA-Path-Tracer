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
	else if (obj.type == MODEL)
	{
		const glm::ivec3& tri = triangles[offset];
		for (int i = 0; i < 3; i++)
		{
			const glm::vec3& vert = vertices[tri[i]];
			const glm::vec3& tmp = glm::vec3(obj.Transform.transform * glm::vec4(vert, 1.0));
			bbox = Union(bbox, tmp);
		}
	}
	assert(bbox.pMin.x < bbox.pMax.x && bbox.pMin.y < bbox.pMax.y && bbox.pMin.z < bbox.pMax.z);
}

BVHNode* buildBVHTreeRecursiveSAH(std::vector<Primitive>& primitives, int start, int end)
{
	BVHNode* root = new BVHNode();
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
	int axis = centerDiff.x >= centerDiff.y && centerDiff.x >= centerDiff.z ? 0 : (centerDiff.y >= centerDiff.x && centerDiff.y >= centerDiff.z ? 1 : 2);
	root->axis = axis;
	root->startPrim = start;
	root->endPrim = end;
	if (numPrims <= MAX_NUM_PRIMS_IN_LEAF || centerDiff[axis] < EPSILON)
	{
		root->left = nullptr;
		root->right = nullptr;
	}
	else
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
		for (int i = 1; i < SAH_BUCKET_SIZE - 1; i++)
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
			root->left = buildBVHTreeRecursiveSAH(primitives, start, mid);
			root->right = buildBVHTreeRecursiveSAH(primitives, mid, end);
		}
		else
		{
			root->left = nullptr;
			root->right = nullptr;
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

int compactBVHTreeForStacklessTraverse(std::vector<BVHGPUNode>& bvhArray, BVHNode* root, int parent)
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
	int left = compactBVHTreeForStacklessTraverse(bvhArray, root->left, curr);
	int right = compactBVHTreeForStacklessTraverse(bvhArray, root->right, curr);
	bvhArray[curr].left = left;
	bvhArray[curr].right = right;
	return curr;
}