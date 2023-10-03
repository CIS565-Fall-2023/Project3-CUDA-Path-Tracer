#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include "tiny_gltf.h"

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

    RenderState state;
};
