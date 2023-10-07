#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include "tiny_gltf.h"

#define BVH 1

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    //load gltf fn
    int loadGltf(string filename, std::vector<Geom>& newGeoms);
    void parseModelNodes(tinygltf::Model& model, tinygltf::Node& node, std::vector<Geom>& newGeoms);
    void parseMesh(tinygltf::Model& model, tinygltf::Mesh& mesh, std::vector<Geom>& newGeoms);

#if BVH
    //sah based bvh + stack iteration mainly adapted from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
    // build bvh tree
    void buildBvhTree();
    //init bvh owned arrs for partitioning
    void initBvhIndexArrs();
    //eval sah
    float eval_sah(BVHNode& node, int axis, float pos, glm::vec3 &l_bb_min, glm::vec3& l_bb_max, 
        glm::vec3 &r_bb_min, glm::vec3 &r_bb_max);
    //subdivide for top down construction
    void subdivide_bvh(BVHNode& node);
    //debug print
    void print_tree(BVHNode& node);
#endif
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    //mesh point data vec
    std::vector<Triangle> mesh_triangles;
    // mapping info vec
    std::vector<ImageInfo> image_infos;
    // image data vec
    std::vector<glm::vec3> image_data;

    //to avoid dup textures
    std::map<string, int> imguri_to_index;

#if BVH
    //BVH vecs
    std::vector<BVHTriIndex> bvh_tri_indices;
    std::vector<BVHGeomIndex> bvh_geom_indices;
    std::vector<BVHNode> bvh_nodes;
#endif

    RenderState state;
};
