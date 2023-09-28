#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Mesh/tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void loadObjGeom(const tinyobj::attrib_t& attrib,
        const std::vector<tinyobj::shape_t>& shapes, std::vector<Geom>& tempTriangles);
    void loadObjMaterial(const std::vector<tinyobj::material_t>& tinyobjMaterials);

    char* filename;
    

    int getLongestAxis(const glm::vec3& minBounds, const glm::vec3& maxBounds);
    float computeBoxSurfaceArea(const glm::vec3& min, const glm::vec3& max);
    int getBestSplit(std::vector<Geom> geoms, int start, int end);
    

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    BVHNode* root;
    std::vector<CompactBVH> bvh;

    BVHNode* constructBVH(std::vector<Geom> geoms, int start, int end);
    int flattenBVHTree(BVHNode* node);
};