#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int parseGenericFile(string filename);
    int parseGLTFModel(const tinygltf::Model &model);
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGLTFPerspectiveCamera(const tinygltf::Camera &cameraObj, glm::vec3& translation);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    RenderState state;
};
