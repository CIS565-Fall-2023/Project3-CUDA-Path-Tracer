#pragma once
#include <vector>
#include "depScene.h"
#include "sceneStructs.h"

class Scene {
    std::vector<Sphere> spheres;

    void applyNodeTransform(const tinygltf::Node& node, glm::mat4x4& parentTransform);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & localTransform);

    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & parentTransform);
    tinygltf::Model model;
public:
    Scene(const char * filename);
    std::vector<Triangle> triangles;
    Triangle* dev_triangles;
    Sphere* dev_spheres;
    Primitive** dev_primitives;
    int getPrimitiveSize() const{
        return triangles.size() + spheres.size();
    }
    void initTriangles();
    void freeBuffer();
};