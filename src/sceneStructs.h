#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"


#define BACKGROUND_COLOR_MULT (3.0f)

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE_MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct ObjectTransform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Object {
    enum GeomType type;
    int materialid;
    int triangleStart, triangleEnd;
    ObjectTransform Transform;
};

struct BoundingBox {
    glm::vec3 pMin, pMax;
    BoundingBox() :pMin(glm::vec3(1e38f)), pMax(glm::vec3(-1e38f)) {}
    glm::vec3 center() const { return (pMin + pMax) * 0.5f; }
};

BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2);
BoundingBox Union(const BoundingBox& b1, const glm::vec3& p);
float BoxArea(const BoundingBox& b);

struct Primitive {
    int objID;
    int offset;//offset for triangles in model
    BoundingBox bbox;
    Primitive(const Object& obj, int objID, int triangleOffset = -1, const glm::ivec3* triangles = nullptr, const glm::vec3* vertices = nullptr);
};

struct BVHNode {
    int axis;
    BVHNode* left, * right;
    int startPrim, endPrim;
    BoundingBox bbox;
};

struct BVHGPUNode
{
    int axis;
    BoundingBox bbox;
    int parent, left, right;
    int startPrim, endPrim;
    BVHGPUNode() :axis(-1), parent(-1), left(-1), right(-1), startPrim(-1), endPrim(-1){}
};



enum MaterialType {
    diffuse, frenselSpecular, microfacet, emitting
};

struct Material {
    glm::vec3 color = glm::vec3(0);
    float indexOfRefraction = 0;
    float emittance = 0;
    float roughness = -1.0;
    cudaTextureObject_t diffuseMap = 0, normalMap = 0;
    MaterialType type = diffuse;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t = -1.0;
    glm::vec3 surfaceNormal = glm::vec3(0.0);
    glm::vec3 worldPos = glm::vec3(0.0);
    int materialId = -1;
    glm::vec2 uv = glm::vec2(0.0);
    MaterialType type = diffuse;
};
