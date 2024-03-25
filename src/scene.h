#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
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
    bool loadModel(const string&, int, bool);
    bool loadGeometry(const string&,int);
    void loadTextureFromFile(const std::string& texturePath, cudaTextureObject_t* texObj, int type);
    void LoadTextureFromMemory(void* data, int width, int height, int bits, int channels, cudaTextureObject_t* texObj);
    void loadSkybox();
public:
    void buildBVH();
    void buildStacklessBVH();
    void LoadAllTextures(); 
    void CreateLights();
    Scene(string filename);
    ~Scene();

    std::vector<Object> objects;
    std::vector<Material> materials;
    std::vector<glm::ivec3> triangles;
    std::vector<glm::vec3> verticies;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<float> fSigns;
    std::vector<Primitive> primitives;
    std::vector<Primitive> lights;
    std::vector<BVHGPUNode> bvhArray;
    std::vector<MTBVHGPUNode> MTBVHArray;
    RenderState state;
    BVHNode* bvhroot = nullptr;
    cudaTextureObject_t skyboxTextureObj = 0;
    int bvhTreeSize = 0;
    std::vector<char*> gltfTexTmpArrays;
    std::vector<cudaArray*> textureDataPtrs;
    std::unordered_map< std::string, cudaTextureObject_t> strToTextureObj;
    std::vector<std::pair<std::string, int> > LoadTextureFromFileJobs;//texture path, materialID
    std::vector <GLTFTextureLoadInfo> LoadTextureFromMemoryJobs;
};

struct MikkTSpaceHelper
{
    Scene* scene;
    int i;
};

struct AliasBin {
    float q, p;
    int alias = -1;
};


