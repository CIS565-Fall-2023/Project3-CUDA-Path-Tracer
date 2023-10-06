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
    void load_mesh(const tinygltf::Model &model, const tinygltf::Mesh& mesh, const glm::vec3& translation, 
                   const glm::vec3& rotation, const glm::vec3& scale, const glm::quat& rotation_quat,
                   const glm::mat4 &transformation = glm::mat4(0.0f));
    void traverse_node(const tinygltf::Model& model, int node_index, const glm::mat4& parent_transform = glm::mat4(1.0f));
    bool load_gltf(string filename);
    bool load_gltf_contents(string filename);
    bool gltf_load_materials(const tinygltf::Model &model);
    std::string basePath;
    std::array<std::string, 2> const supported_attributes = {
        "POSITION",
        "NORMAL"
        //TEXCOORD?
    };
    std::map<int, int> material_map;
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
    uint32_t nodes_used = 0;
};
