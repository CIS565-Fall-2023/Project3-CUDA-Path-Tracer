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
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadOBJ(string filePath, Geom& mesh);

    void BuildBVH();
    void UpdateNodeBounds(int nodeIdx);
    void Subdivide(int nodeIdx);

public:
    Scene(string filename);
    ~Scene();

    bool hasMesh = false;
    int N; // num of triangles
    std::vector<BVHNode> bvhNode;
    int rootNodeIdx = 0, nodesUsed = 1;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tri;
    std::vector<int> triIdx;
    RenderState state;
};
