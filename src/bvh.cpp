#include "bvh.h"

void BVHAccel::buildBVH(std::vector<Triangle> prims)
{
	std::stack<BVHNodeInfo *> nodeStack;
	orderedPrims = prims;
	root = new BVHNodeInfo();
	root->startPrim = 0;
	root->endPrim = prims.size();	
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
		printf("index: %d startPrim: %d endPrim: %d\n", index, currNodeInfo->startPrim, currNodeInfo->endPrim);
		printf("isLeft: %d\n", currNodeInfo->isLeft);
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
			printf("dim: %d\n", dim);
			printf("aabb: %f %f %f %f %f %f\n", aabb.pmin.x, aabb.pmin.y, aabb.pmin.z, aabb.pmax.x, aabb.pmax.y, aabb.pmax.z);
			std::sort(orderedPrims.begin() + currNodeInfo->startPrim, orderedPrims.begin() + currNodeInfo->endPrim, [dim](const Triangle& a, const Triangle& b) {
				return getAABB(a).centroid[dim] < getAABB(b).centroid[dim];
			});
			for (size_t i = currNodeInfo->startPrim; i < currNodeInfo->endPrim; i++)
			{
				auto aabb = getAABB(orderedPrims[i]);
				printf("i : %d\n", i);
				printf("prim: %f %f %f\n", aabb.centroid[0], aabb.centroid[1], aabb.centroid[2]);
			}

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
			
			printf("leftNode: %d %d\n", leftNode->startPrim, leftNode->endPrim);
			printf("rightNode: %d %d\n", rightNode->startPrim, rightNode->endPrim);
			
			nodeStack.push(rightNode);
			nodeStack.push(leftNode);
		}
		index++;
	}
	nodeCount = index;
	serializeBVH();
}

void BVHAccel::traverseBVHNonSerialized()
{
	std::stack<BVHNodeInfo *> nodeStack;
	nodeStack.push(root);
	while (nodeStack.size() != 0) {
		BVHNodeInfo * currNodeInfo = nodeStack.top();
		nodeStack.pop();
		if (currNodeInfo->right) nodeStack.push(currNodeInfo->right);
		if (currNodeInfo->left)  nodeStack.push(currNodeInfo->left);

	}
}

void BVHAccel::serializeBVH()
{
	std::stack<BVHNodeInfo*> nodeStack;
	nodeStack.push(root);
	serializedNodeInfos.resize(nodeCount);
	while (nodeStack.size() != 0) {
		BVHNodeInfo* currNodeInfo = nodeStack.top();
		BVHNode node;
		node.aabb = currNodeInfo->aabb;
		node.left = currNodeInfo->left ? currNodeInfo->left->index : -1;
		node.right = currNodeInfo->right ? currNodeInfo->right->index : -1;
		node.parent	= currNodeInfo->parent ? currNodeInfo->parent->index : -1;
		node.hit = currNodeInfo->index + 1;
		if (currNodeInfo->isLeaf) {
			node.miss = node.hit;
		}
		else if (currNodeInfo->isLeft && currNodeInfo->parent) {
			node.miss = currNodeInfo->parent->right->index;
		}
		else {
			node.miss = currNodeInfo->index + getSubTreeSize(currNodeInfo);
		}
		serializedNodeInfos[currNodeInfo->index] = currNodeInfo;
		printf("\n");
		printf("node.index: %d\n", currNodeInfo->index);
		printf("node.hit: %d\n", node.hit);
		printf("node.miss: %d\n", node.miss);
		printf("node.left: %d\n", node.left);
		printf("node.right: %d\n", node.right);
		printf("node.parent: %d\n", node.parent);
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

bool intersectAABB(const Ray& ray, const AABB& aabb)
{
	glm::vec3 invDirection = 1.0f / ray.direction;

	float t1 = (aabb.pmin.x - ray.origin.x) * invDirection.x;
	float t2 = (aabb.pmax.x - ray.origin.x) * invDirection.x;
	float t3 = (aabb.pmin.y - ray.origin.y) * invDirection.y;
	float t4 = (aabb.pmax.y - ray.origin.y) * invDirection.y;
	float t5 = (aabb.pmin.z - ray.origin.z) * invDirection.z;
	float t6 = (aabb.pmax.z - ray.origin.z) * invDirection.z;

	float tmin = glm::max(glm::max(glm::min(t1, t2), glm::min(t3, t4)), glm::min(t5, t6));
	float tmax = glm::min(glm::min(glm::max(t1, t2), glm::max(t3, t4)), glm::max(t5, t6));

	return tmax >= tmin && tmax >= 0;
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
