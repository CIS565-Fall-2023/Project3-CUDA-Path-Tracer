#pragma once

#include "sceneStructs.h"
#include "utilities.h"

struct BVHNode
{
    // Based on PBRT: https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
    AABB bounds;
    //BVHNode* left;
    //BVHNode* right;

    int leftChildIdx;
    int rightChildIdx;
    int splitAxis;
    int triIdx;
    int nodeIdx;

    std::vector<int> meshIds;

    void initAsLeafNode(int triIdx, const AABB& bounds)
    {
        this->bounds = bounds;
        this->triIdx = triIdx;
        this->leftChildIdx = -1;
        this->rightChildIdx = -1;
    }

    void initInterior(int splitAxis, int leftChildIdx, int rightChildIdx, const std::vector<BVHNode>& bvhNodes)
    {
        this->leftChildIdx = leftChildIdx;
        this->rightChildIdx = rightChildIdx;
        bounds = AABB::combine(bvhNodes[this->leftChildIdx].bounds, bvhNodes[this->leftChildIdx].bounds);
        this->splitAxis = splitAxis;
        this->triIdx = -1;  // no triangle here
    }
};

class BVH
{
private:
    std::vector<uPtr<BVHNode>> nodes;
	BVHNode* root;
    int rootNodeIdx;

public:
    BVH();

    std::vector<Triangle*> orderedTris;

    int buildBVHRecursively(int& totalNodes, int startOffset, int nTris, const std::vector<Triangle>& tris, std::vector<int>& triIdices, std::vector<BVHNode>& bvhNodes);
    const BVHNode* getRootNode() const;
};

struct sortTriIndicesBasedOnDim
{
private:
    const std::vector<Triangle>& tris;
    int dimToSortOn;
public:
    sortTriIndicesBasedOnDim(const std::vector<Triangle>& tris, int dimToSortOn) : tris(tris), dimToSortOn(dimToSortOn) {}
    bool operator()(int i, int j) const { return tris[i].centroid[dimToSortOn] < tris[j].centroid[dimToSortOn]; }
};

//bool xSort(Triangle* a, Triangle* b) { return a->centroid.x < b->centroid.x; }
//bool ySort(Triangle* a, Triangle* b) { return a->centroid.y < b->centroid.y; }
//bool zSort(Triangle* a, Triangle* b) { return a->centroid.z < b->centroid.z; }