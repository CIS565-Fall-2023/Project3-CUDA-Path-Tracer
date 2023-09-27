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
    void loadTexture(const std::string& texturePath, cudaTextureObject_t* texObj, int type);
    void loadSkybox();
public:
    void buildBVH();
    void buildStacklessBVH();
    void LoadAllTextures();
    Scene(string filename);
    ~Scene();

    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<glm::ivec3> triangles;
    std::vector<glm::vec3> verticies;
    std::vector<glm::vec2> uvs;
    std::vector<Primitive> primitives;
    std::vector<BVHGPUNode> bvhArray;
    RenderState state;
    BVHNode* bvhroot = nullptr;
    cudaTextureObject_t skyboxTextureObj = 0;
    std::vector<cudaArray*> textureDataPtrs;
    std::vector<std::pair<std::string, int> > textureLoadJobs;//texture path, materialID
};
