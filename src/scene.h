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
    std::unordered_map<string, SceneMeshGroup> loadedMeshGroups;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    SceneMeshGroup loadGltfMesh(string path);
    /// <summary>
    /// Recursively parses the entire node. Some nodes have children which is why the recursion is necessary.
    /// Calculates the AABB of the SceneMeshGroup generated here.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="node"></param>
    int parseGltfNodeRecursive(const tinygltf::Model& model, const tinygltf::Node& node, glm::vec3& aabbMin, glm::vec3& aabbMax);
    /// <summary>
    /// Does the actual parsing from the recursive function.
    /// Calculates the AABB for the SceneMesh generated here.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="node"></param>
    int parseGltfNodeHelper(const tinygltf::Model& model, const tinygltf::Node& node, glm::vec3& aabbMin, glm::vec3& aabbMax);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tris;
    std::vector<SceneMesh> meshes;

    RenderState state;
};
