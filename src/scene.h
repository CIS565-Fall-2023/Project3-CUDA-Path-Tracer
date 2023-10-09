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
    //std::unordered_map<string, uPtr<BVH>> meshBVHs;

    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    SceneMeshGroup loadGltfMesh(string path);
    /// <summary>
    /// Recursively parses the entire node. Some nodes have children which is why the recursion is necessary.
    /// Calculates the AABB of the SceneMeshGroup generated here.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="node"></param>
    int parseGltfNodeRecursive(const tinygltf::Model& model, const tinygltf::Node& node, AABB& aabb);
    /// <summary>
    /// Does the actual parsing from the recursive function.
    /// Calculates the AABB for the SceneMesh generated here.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="node"></param>
    int parseGltfNodeHelper(const tinygltf::Model& model, const tinygltf::Node& node, AABB& aabb);
    int loadCamera();

    /// <summary>
    /// Returns this geom's root node index for the BVH
    /// </summary>
    /// <param name="meshPath"></param>
    /// <param name="startTriIdx"></param>
    /// <param name="endTriIdx"></param>
    /// <returns></returns>
    int constructBVH(string meshPath, unsigned int startTriIdx, unsigned int endTriIdx);
    int buildBVHRecursively(int& totalNodes, int startOffset, int nTris, const std::vector<Triangle>& tris, std::vector<int>& triIndices, std::vector<BVHNode>& bvhNodes);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> tris;
    std::vector<SceneMesh> meshes;
    std::vector<BVHNode> bvhNodes;

    RenderState state;
};
