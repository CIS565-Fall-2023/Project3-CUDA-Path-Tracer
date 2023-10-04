#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <unordered_map>

using namespace std;

class Scene {
private:
    string basePath;

    ifstream fp_in;
    int loadMaterial(string materialId);
    int loadGeom(string objectid);
    int loadCamera();

    int loadMesh(string filePath, int& startTriIdx, int& numTris);
    int buildBvh(int startTri, int numTris);
    void bvhUpdateNodeBounds(BvhNode& node);
    float bvhEvaluateSAH(BvhNode& node, int axis, float pos);
    float bvhFindBestSplitPlane(BvhNode& node, int& axis, float& splitPos, AABB& leftChildBox, AABB& rightChildBox);
    void bvhSubdivide(BvhNode& node);

    int loadTexture(string filePath, bool isNormals = false);

    unordered_map<string, int> bvhRootIndices;
    unordered_map<string, int> textureIndices;

    unordered_map<int, int> geomIdMap;

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<Triangle> tris;
    std::vector<BvhNode> bvhNodes;
    RenderState state;
};
