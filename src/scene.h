#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "tinygltf/tiny_gltf.h"

#include "sceneStructs.h"
#include "bsdfStruct.h"

class Scene {
    std::vector<Sphere> spheres;

    void applyNodeTransform(const tinygltf::Node& node, glm::mat4x4& parentTransform);
    void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & localTransform);

    void processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & parentTransform);
    tinygltf::Model model;
public:
    Scene(const char * filename);
    std::vector<Triangle> triangles;
    std::vector<BSDFStruct> bsdfStructs;
    Triangle* dev_triangles;
    Sphere* dev_spheres;
    Primitive** dev_primitives;
    int getPrimitiveSize() const{
        return triangles.size() + spheres.size();
    }
    void initTriangles();
    void initBSDFs();
};



