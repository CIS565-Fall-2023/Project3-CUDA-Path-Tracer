#pragma once
#include "glm/glm.hpp"
#include <vector>
#include "sceneStructs.h"


BVHNode* buildBVHTreeRecursiveSAH(std::vector<Primitive>& primitives, int start, int end, int* size);
void destroyBVHTree(BVHNode* root);
int recursiveCompactBVHTreeForStacklessTraverse(std::vector<BVHGPUNode>& bvhArray, BVHNode* root, int parent = -1);
bool checkBVHTreeFull(BVHNode* root);
void compactBVHTreeToMTBVH(std::vector<MTBVHGPUNode>& MTBVHGPUNode, BVHNode* root, int treeSize);
