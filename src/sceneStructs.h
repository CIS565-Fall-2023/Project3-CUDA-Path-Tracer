
#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <array>
#include <iostream>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType 
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray 
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom 
{
    enum GeomType type;
    int materialId;
    int referenceId; // for now, used only for meshes
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material 
{
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Mesh
{
    int bvhRootNode; // TODO: remove Mesh and have Geom store this instead
};

struct Vertex
{
    glm::vec3 pos;
};

struct Triangle
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
    glm::vec3 centroid;
};

struct AABB
{
    glm::vec3 bMin = glm::vec3(FLT_MAX), bMax = glm::vec3(-FLT_MAX);
    __host__ __device__ void grow(const glm::vec3& p)
    {
        bMin = glm::min(bMin, p);
        bMax = glm::max(bMax, p);
    }
    __host__ __device__ void grow(const Triangle& tri)
    {
        grow(tri.v0.pos);
        grow(tri.v1.pos);
        grow(tri.v2.pos);
    }
    __host__ __device__ glm::vec3 extent()
    {
        return bMax - bMin;
    }
    __host__ __device__ float surfaceArea()
    {
        glm::vec3 extent = this->extent();
        return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
    }
};

struct BvhNode
{
    AABB aabb;
    int leftFirst, triCount;
    __host__ __device__ bool isLeaf() const { return triCount > 0; }

    __host__ friend std::ostream& operator<<(std::ostream& os, const BvhNode& node)
    {
        os << "bounding box: [" << node.aabb.bMin.x << ", " << node.aabb.bMin.y << ", " << node.aabb.bMin.z << "] - ["
            << node.aabb.bMax.x << ", " << node.aabb.bMax.y << ", " << node.aabb.bMax.z << "]" << std::endl;
        os << "leftFirst: " << node.leftFirst << ", triCount: " << node.triCount;
        return os;
    }
};

struct Camera 
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDistance;
};

struct RenderState 
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment 
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int bouncesSoFar;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection 
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;

    __host__ __device__ bool operator<(const ShadeableIntersection& other) const {
        return materialId < other.materialId;
    }
};
