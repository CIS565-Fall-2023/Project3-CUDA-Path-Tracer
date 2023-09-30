#pragma once
#include "glm/glm.hpp"
#include <vector>
#include <stack>
#include <algorithm>
#include "sceneStructs.h"

#define BVH_LOG 0
#define BVH_NAIVE 1
#define BVH_SAH 0

struct AABB {
	glm::vec3 pmax;
	glm::vec3 pmin;
	glm::vec3 centroid;
};

AABB unionAABB(const AABB &box1, const AABB &box2);
AABB getAABB(const Triangle &tri);
int getLongestAxis(const AABB &box);
float getArea(const AABB &box);

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
	bool isLeaf;
};

class BVHAccel {
	BVHNodeInfo* root;
	std::vector<BVHNodeInfo*> serializedNodeInfos;
	int nodeCount = 0;
	const int maxPrimitivesInNode = 4;
	void buildBVH();
	void serializeBVH();
	//void traverseBVHNonSerialized();
public:
	std::vector<Triangle> orderedPrims;
	std::vector<BVHNode> nodes;
	void initBVH(std::vector<Triangle> prims);
	/* Transforming BVH from tree to array */
};



