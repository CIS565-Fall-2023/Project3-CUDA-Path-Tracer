#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "bvh.h"
using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadObject(string objectid);
    int loadCamera();
    bool loadModel(const string&, int);
    bool loadGeometry(const string&,int);
    
public:
    void buildBVH();
    void buildStacklessBVH();
    Scene(string filename);
    ~Scene();

    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<glm::ivec3> triangles;
    std::vector<glm::vec3> verticies;
    std::vector<Primitive> primitives;
    std::vector<BVHGPUNode> bvhArray;
    RenderState state;
    BVHNode* bvhroot = nullptr;
};
