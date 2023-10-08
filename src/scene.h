#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "bound.h"
#include "geom.h"
#include "Mesh/tiny_obj_loader.h"

#define USE_BVH 1
#define DEBUG_BVH 0

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

    int addObjMaterial(const tinyobj::material_t& mat);
    int loadObj(const string& objFilePath, int materialid, 
        const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale,
        const glm::mat4& transform, const glm::mat4& inverseTransform, 
        const glm::mat4& invTranspose);

    int partitionSplit(std::vector<BVHGeomInfo>& geomInfo, int start, int end, int splitAxis, int geomCount,
        Bound& centroidBounds, Bound& bounds);

    BVHNode* constructBVHTree(std::vector<BVHGeomInfo>& geomInfo, int start, int end,
        int* totalNodes, std::vector<Geom>& orderedGeoms);
    int flattenBVHTree(BVHNode* node, int* offset);
    
    string filename;
    int LEAF_SIZE, nBuckets;

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    int sqrtSamples;

    std::vector<LinearBVHNode> bvh;
    void buildBVH();
};