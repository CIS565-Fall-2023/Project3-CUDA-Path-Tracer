#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

// #define MATERIAL_SORT
#define CACHE_FIRST_BOUNCE
// #define JITTER_RAY
// #define DEBUG_OUTPUT
#define USING_BVH

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void buildTree();
    void splitTree(int left, int right, int bbox, int axis);
    void printTree();
    bool checkTree();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tris;
    std::vector<BoundingBox> bvh;

    RenderState state;
    int objCount = 0;
};
