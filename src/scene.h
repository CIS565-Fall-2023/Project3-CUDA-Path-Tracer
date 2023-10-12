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

    int envMap;
    int numLights;
    std::vector<Geom> lights;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tris;
    std::vector<Image> textures;
    RenderState state;
};
