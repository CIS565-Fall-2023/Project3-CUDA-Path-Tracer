#pragma once
#include <vector>
#include "scene.h"
#include "sceneStructs.h"

class PrimitiveAssmbler {
    std::vector<Sphere> spheres;

    void applyNodeTransform(const tinygltf::Node& node, glm::mat4x4& parentTransform);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & localTransform);

    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & parentTransform);

public:
    std::vector<Triangle> triangles;
    void movePrimitivesToDevice();
    Triangle* dev_triangles;
    Sphere* dev_spheres;
    Primitive** dev_primitives;
    int getPrimitiveSize() const{
        return triangles.size() + spheres.size();
    }
    void assembleScenePrimitives(Scene* scene);
    void freeBuffer();
};