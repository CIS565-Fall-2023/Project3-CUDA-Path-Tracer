#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    string basePath;
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadMeshGltf(string filename, Geom& geom, int objectId);
    int loadMeshObj(string filename, Geom& geom);

    void buildBVH();
    void updateNodeBounds(int nodeIdx);
    void subdivide(int nodeIdx);

    void updateBounds(const int idx);
    void chooseSplit(BVHNode* node, float& split, int& axis);
    void addChildren(BVHNode* node);
    void generateBVH();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<int> triIndices;
    std::vector<BVHNode> bvhNodes;
    std::vector<AABB> geomAABBs;
    int nodesUsed = 1; // keep track of the current available node in the tree
    int meshCount;
    RenderState state;
};
