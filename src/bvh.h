#pragma once
#include "glm/glm.hpp"
#include <vector>
#include "sceneStructs.h"

#define MAX_NUM_PRIMS_IN_LEAF 2
#define SAH_BUCKET_SIZE 12
#define SAH_RAY_BOX_INTERSECTION_COST 0.1f


BVHNode* buildBVHTreeRecursiveSAH(std::vector<Primitive>& primitives, int start, int end);
void destroyBVHTree(BVHNode* root);
int compactBVHTreeForStacklessTraverse(std::vector<BVHGPUNode>& bvhArray, BVHNode* root, int parent = -1);
bool checkBVHTreeFull(BVHNode* root);
