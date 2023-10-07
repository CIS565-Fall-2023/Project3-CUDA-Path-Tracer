#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    std::unordered_map<string, int> loadedMeshes;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadGltfMesh(string path);
    /// <summary>
    /// Recursively parses the entire node. Some nodes have children which is why the recursion is necessary.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="node"></param>
    void parseGltfNode(const tinygltf::Model& model, const tinygltf::Node& node);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
