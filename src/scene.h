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
    int loadGeom(string objectid);
    int loadCamera();
    int loadObj(const char* inputfile);
    int loadTexture(string textureID);
public:
    Scene(string filename);
    ~Scene();

    BVHNode* buildBVH(int start_index, int end_index, int& level, int count);
    glm::vec3 getBounds(int start_index, int end_index, bool is_max, bool is_centroid);
    void reformatBVHToGPU();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;
    std::vector<glm::vec3> textureColors;
    int triIdx = 0;
    RenderState state;

    BVHNode* root_node;
    int num_nodes = 0;

    std::vector<BVHNode_GPU> bvh_nodes_gpu;
    std::vector<TriBounds> tri_bounds;
    std::vector<Triangle> mesh_tris_sorted;

    std::vector<Geom> lights;
    int lightMaterialNum;
    int lightNum = 0;
    int num_tris = 0;
};
