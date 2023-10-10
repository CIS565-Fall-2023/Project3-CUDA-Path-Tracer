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
    int parseGenericFile(string filename);
    int parseGLTFModel(const tinygltf::Model &model);
    int parseGLTFNode(const int node, const tinygltf::Model &model, glm::mat4& baseTransform);
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGLTFPerspectiveCamera(const tinygltf::Camera &cameraObj, glm::vec3& translation);
    Geom findBoundingVolume(float* vertices, int numVertices);
    int addGlobalIllumination();
    int addDefaultCamera(glm::mat4& transform);
    Octree buildOctree(const Mesh& mesh);
    template <typename Iterator>
    int buildOctreeImpl(Octree& tree, const Geom& boundingBox, int depth,  Iterator begin, Iterator end);
    Geom getAxisAlignedBoundingBox(const Geom& meshBoundingVolume);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Mesh> meshes;
    std::vector<Octree> octrees;
    std::vector<Material> materials;
    RenderState state;
};
