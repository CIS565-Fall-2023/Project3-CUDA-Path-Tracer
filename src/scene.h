#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    string scene_filename;
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Vertex> vertices;
    // std::vector<float> vertices;  // 'v'
    // std::vector<float> normals;   // 'vn'
    // std::vector<float> texcoords; // 'vt'
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::unordered_map<string, int> material_map;
    RenderState state;
};
