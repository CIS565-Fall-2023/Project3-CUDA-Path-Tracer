
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
    int bvhRootNodeIdx; // used only for meshes
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material 
{
    struct {
        glm::vec3 color = glm::vec3(1, 0, 1);
        int textureIdx = -1;
    } diffuse;
    struct {
        float exponent = 0;
        glm::vec3 color = glm::vec3(0);
        float hasReflective = 0;
        float hasRefractive = 0;
        float indexOfRefraction = 1.45;
    } specular;
    struct {
        glm::vec3 color = glm::vec3(0);
        float strength = 0;
    } emission;
    struct
    {
        int textureIdx = -1;
    } normalMap;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
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
    __host__ __device__ void grow(const AABB& other)
    {
        grow(other.bMin);
        grow(other.bMax);
    }
    __host__ __device__ glm::vec3 extent() const
    {
        return bMax - bMin;
    }
    __host__ __device__ float surfaceArea() const
    {
        glm::vec3 extent = this->extent();
        return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
    }
};

struct BvhNode
{
    AABB aabb; // 6 floats = 24 bytes
    int leftFirst, triCount; // 2 ints = 8 bytes
    __host__ __device__ bool isLeaf() const { return triCount > 0; }
    __host__ __device__ float cost() const { return triCount * aabb.surfaceArea(); }

#if DEBUG_PRINT_BVH
    __host__ friend std::ostream& operator<<(std::ostream& os, const BvhNode& node)
    {
        os << "bounding box: [" << node.aabb.bMin.x << ", " << node.aabb.bMin.y << ", " << node.aabb.bMin.z << "] - ["
            << node.aabb.bMax.x << ", " << node.aabb.bMax.y << ", " << node.aabb.bMax.z << "]" << std::endl;
        os << "leftFirst: " << node.leftFirst << ", triCount: " << node.triCount;
        return os;
    }
#endif
};

struct Texture
{
    unsigned char* host_dataPtr;
    int width;
    int height;
    int channels;
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
    float focusDistance;
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
    int hitGeomIdx;
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    int materialId;
    int triIdx;

    __host__ __device__ bool operator<(const ShadeableIntersection& other) const {
        return materialId < other.materialId;
    }
};
