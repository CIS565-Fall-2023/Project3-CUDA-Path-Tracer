#pragma once
#include "glm/glm.hpp"
#include <vector>
#include <stack>
#include <algorithm>
#include "sceneStructs.h"

struct AABB {
	glm::vec3 pmax;
	glm::vec3 pmin;
	glm::vec3 centroid;
};

AABB unionAABB(const AABB &box1, const AABB &box2);
AABB getAABB(const Triangle &tri);
int getLongestAxis(const AABB &box);
float getArea(const AABB &box);

bool intersectAABB(const Ray &ray, const AABB &box);

struct BVHNodeInfo {
	AABB aabb;
	BVHNodeInfo * left = nullptr;
	BVHNodeInfo * right = nullptr;
	BVHNodeInfo* parent = nullptr;
	int hitIndex;
	int missIndex;
	int index;
	int startPrim;
	int endPrim;
	int axis;
	bool isLeft = false;
	bool isLeaf = false;
};

int getSubTreeSize(BVHNodeInfo* node);

struct BVHNode {
	AABB aabb;
	int left;
	int right;
	int parent;
	int startPrim;
	int endPrim;
	int hit;
	int miss;
};

class BVHAccel {
	BVHNodeInfo* root;
	std::vector<BVHNodeInfo*> serializedNodeInfos;
	int nodeCount = 0;
public:
	std::vector<Triangle> orderedPrims;
	std::vector<BVHNode> nodes;
	void buildBVH(std::vector<Triangle> prims);

	void traverseBVHNonSerialized();
	/* Transforming BVH from tree to array */
	void serializeBVH();
};


