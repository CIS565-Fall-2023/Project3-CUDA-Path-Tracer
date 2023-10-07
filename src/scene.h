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
    int loadMesh(string meshid);
    int loadEnv(string envid);
public:
    Scene(string filename);
    ~Scene();

    bool hasEnvMap;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tris;
    EnvironmentMap mp;
    RenderState state;
};
