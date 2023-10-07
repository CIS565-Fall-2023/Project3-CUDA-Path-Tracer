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
    int loadTexture(string textureid);
    int loadGeom(string objectid);
    int loadCamera();
    bool loadGLTF(const std::string& filePath, GLTFMesh& gltfMesh, glm::vec3 trans, glm::vec3 scale);
    bool loadOBJ(const std::string& filePath, GLTFMesh& gltfMesh);
public:
    Scene(string filename);
    ~Scene();

    std::vector<GLTFMesh> gltfMeshes;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<glm::vec3> texData;
    RenderState state;
};
