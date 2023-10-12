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

    void loadObj(Geom& newGeom, string obj_filename);
    void loadGltf(Geom& newGeom, string gltf_filename);
    int buildBVH(int meshStartIdx, int meshEndIdx);

public:
    Scene(string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<glm::vec3> vertices;  // 'v'
    std::vector<glm::vec3> normals;   // 'vn'
    std::vector<glm::vec2> texcoords; // 'vt'
    std::vector<Mesh> meshes;
    std::vector<BVHNode> bvh;
    std::vector<Material> materials;
    std::unordered_map<string, int> material_map;
    RenderState state;
};
