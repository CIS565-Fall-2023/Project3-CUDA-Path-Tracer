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

    // Mesh loading
    int loadMeshGltf(string filename, Geom& geom, int objectId);
    int loadMeshObj(string filename, Geom& geom);

    // BVH optimization
    void updateNodeBounds(int nodeIdx);
    void subdivide(BVHNode* node);
    float evaluateSAH(BVHNode* node, float query, int axis);
    void buildBVH();

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<int> triIndices;
    std::vector<BVHNode> bvhNodes;
    int nodesUsed = 1; // keep track of the current available node in the tree
    int meshCount;
    RenderState state;
};
