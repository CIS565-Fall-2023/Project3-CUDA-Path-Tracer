#include "bvh.h"


void BVHAccel::initBVH(std::vector<Triangle> prims)
{
	orderedPrims = prims;
	buildBVH();
	serializeBVH();
}


void BVHAccel::buildBVH()
{
	std::stack<BVHNodeInfo *> nodeStack;
	root = new BVHNodeInfo();
	root->startPrim = 0;
	root->endPrim = orderedPrims.size();
	root->isLeft = false;
	nodeStack.push(root);
	BVHNodeInfo * currNodeInfo;
	int index = 0;
	while (nodeStack.size() != 0) {
		// Pop the current node
		currNodeInfo = nodeStack.top();
		nodeStack.pop();
		currNodeInfo->index = index;
		AABB aabb;
#if BVH_LOG
		printf("index: %d startPrim: %d endPrim: %d\n", index, currNodeInfo->startPrim, currNodeInfo->endPrim);
		printf("isLeft: %d\n", currNodeInfo->isLeft);
#endif
		for (size_t i = currNodeInfo->startPrim; i < currNodeInfo->endPrim; i++)
		{
			aabb = unionAABB(aabb, getAABB(orderedPrims[i]));
		}
		if (currNodeInfo->endPrim - currNodeInfo->startPrim < 4) {
			currNodeInfo->isLeaf = true;
			currNodeInfo->aabb = aabb;
		}
		else {

			// Split the bounding box
		
			int dim = getLongestAxis(aabb);
			std::sort(orderedPrims.begin() + currNodeInfo->startPrim, orderedPrims.begin() + currNodeInfo->endPrim, [dim](const Triangle& a, const Triangle& b) {
				return getAABB(a).centroid[dim] < getAABB(b).centroid[dim];
			});
#if BVH_LOG
			printf("dim: %d\n", dim);
			printf("aabb: %f %f %f %f %f %f\n", aabb.pmin.x, aabb.pmin.y, aabb.pmin.z, aabb.pmax.x, aabb.pmax.y, aabb.pmax.z);
			for (size_t i = currNodeInfo->startPrim; i < currNodeInfo->endPrim; i++)
			{
				auto aabb = getAABB(orderedPrims[i]);
				printf("i : %d\n", i);
				printf("prim: %f %f %f\n", aabb.centroid[0], aabb.centroid[1], aabb.centroid[2]);
			}
#endif
			/* Use binary split for now */
			int mid = (currNodeInfo->startPrim + currNodeInfo->endPrim) / 2;
			// TODO: More heuristics!

			BVHNodeInfo * leftNode  = new BVHNodeInfo();
			leftNode->parent = currNodeInfo;
			leftNode->startPrim = currNodeInfo->startPrim;
			leftNode->endPrim = mid;
			leftNode->isLeft = true;


			BVHNodeInfo * rightNode = new BVHNodeInfo();
			rightNode->parent = currNodeInfo;
			rightNode->startPrim = mid;
			rightNode->endPrim = currNodeInfo->endPrim;
			rightNode->isLeft = false;
			currNodeInfo->left = leftNode;
			currNodeInfo->right = rightNode;
			currNodeInfo->isLeaf = false;
			currNodeInfo->aabb = aabb;
#if BVH_LOG
			printf("leftNode: %d %d\n", leftNode->startPrim, leftNode->endPrim);
			printf("rightNode: %d %d\n", rightNode->startPrim, rightNode->endPrim);
#endif
			nodeStack.push(rightNode);
			nodeStack.push(leftNode);
		}
		index++;
	}
	nodeCount = index;
	serializeBVH();
}

void BVHAccel::serializeBVH()
{
	std::stack<BVHNodeInfo*> nodeStack;
	nodeStack.push(root);
	nodes.resize(nodeCount);
	while (nodeStack.size() != 0) {
		BVHNodeInfo* currNodeInfo = nodeStack.top();
		BVHNode node;
		node.aabb = currNodeInfo->aabb;
		node.left = currNodeInfo->left ? currNodeInfo->left->index : -1;
		node.right = currNodeInfo->right ? currNodeInfo->right->index : -1;
		node.parent	= currNodeInfo->parent ? currNodeInfo->parent->index : -1;
		node.hit = currNodeInfo->index + 1;
		node.startPrim = currNodeInfo->startPrim;
		node.endPrim = currNodeInfo->endPrim;
		if (currNodeInfo->isLeaf) {
			node.miss = node.hit;
		}
		else if (currNodeInfo->isLeft && currNodeInfo->parent) {
			node.miss = currNodeInfo->parent->right->index;
		}
		else {
			node.miss = currNodeInfo->index + getSubTreeSize(currNodeInfo);
		}
		node.isLeaf = currNodeInfo->isLeaf;
		nodes[currNodeInfo->index] = node;
#if BVH_LOG
		printf("\n");
		printf("node.index: %d\n", currNodeInfo->index);
		printf("node.hit: %d\n", node.hit);
		printf("node.miss: %d\n", node.miss);
		printf("node.left: %d\n", node.left);
		printf("node.right: %d\n", node.right);
		printf("node.parent: %d\n", node.parent);
#endif
		nodeStack.pop();
		if (currNodeInfo->right) nodeStack.push(currNodeInfo->right);
		if (currNodeInfo->left)  nodeStack.push(currNodeInfo->left);
	}
}

AABB unionAABB(const AABB& box1, const AABB& box2)
{
	AABB box;
	box.pmin = glm::min(box1.pmin, box2.pmin);
	box.pmax = glm::max(box1.pmax, box2.pmax);
	box.centroid = (box.pmax + box.pmin) / 2.0f;
	return box;
}

AABB getAABB(const Triangle& tri)
{
	AABB box;
	box.pmin = glm::min(glm::min(tri.p1, tri.p2), tri.p3);
	box.pmax = glm::max(glm::max(tri.p1, tri.p2), tri.p3);
	box.centroid = (box.pmax + box.pmin) / 2.0f;
	return box;
}

int getLongestAxis(const AABB& box)
{
	float x = box.pmax.x - box.pmin.x;
	float y = box.pmax.y - box.pmin.y;
	float z = box.pmax.z - box.pmin.z;
	if (x > y && x > z) {
		return 0;
	}
	else if (y > x && y > z) {
		return 1;
	}
	else {
		return 2;
	}
}

float getArea(const AABB& box)
{
	return (box.pmax.x - box.pmin.x) * (box.pmax.y - box.pmin.y) * (box.pmax.z - box.pmin.z);
}

int getSubTreeSize(BVHNodeInfo* node)
{
	if (node == nullptr) return 0;
	BVHNodeInfo* nodeTmp = node;
	while (nodeTmp->right) {
		nodeTmp = nodeTmp->right;
	}
	return nodeTmp->index - node->index + 1;
}
