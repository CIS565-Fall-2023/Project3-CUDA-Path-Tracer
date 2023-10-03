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
  
    int partitionSplit(std::vector<BVHGeomInfo>& geomInfo, int start, int end, int splitAxis, int geomCount,
        const glm::vec3& centroidMin, const glm::vec3& centroidMax);

    BVHNode* constructBVHTree(std::vector<BVHGeomInfo>& geomInfo, int start, int end,
        int* totalNodes, std::vector<Geom> orderedGeoms);
    int flattenBVHTree(BVHNode* node);
    

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<CompactBVH> bvh;
    void buildBVH();
};