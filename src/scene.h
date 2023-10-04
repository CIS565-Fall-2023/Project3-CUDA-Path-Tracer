#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "texture_fetch_functions.h"
#include "texture.cuh"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadMesh(string objectid);
    int loadTexture(string textureid);
    int loadEnvMap();
public:
    Scene(string filename);
    //~Scene();
    std::vector<Mesh> meshs;
    std::vector<Triangle> trigs;
    std::vector<Material> materials;
    std::vector<CudaTexture> texs;
    int envTexId = -1;
    RenderState state;
};
