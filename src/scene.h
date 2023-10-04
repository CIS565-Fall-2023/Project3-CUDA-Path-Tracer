#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tinygltf/tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    void bvh_update_node_bounds(uint32_t node_index);
    void bvh_build();
    void bvh_subdivide(uint32_t node_index);
    float bvh_find_best_split(uint32_t node_index, int &axis, float &split_pos);
    void bvh_reorder_tris();
    bool bvh_in_use = false;
    bool load_gltf(string filename);
    bool gltf_load_materials(const tinygltf::Model &model);
    std::string basePath;
public:
    Scene(string filename);
    ~Scene();

    bool using_bvh();
    std::vector<Geom> geoms;
    std::vector<Triangle> tris;
    std::vector<int> tri_indices;
    std::vector<Material> materials;
    RenderState state;
    std::vector<BvhNode> bvh_nodes;
    uint32_t root_node_index = 0;
    uint32_t nodes_used = 0;
};
