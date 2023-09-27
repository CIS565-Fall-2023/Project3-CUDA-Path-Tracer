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
    int loadMesh(string objectid);
public:
    Scene(string filename);
    //~Scene();
    std::vector<Mesh> meshs;
    std::vector<Triangle> trigs;
    std::vector<Material> materials;
    RenderState state;
};
