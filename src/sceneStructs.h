#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE,
    OBJ
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct BBox
{
    glm::vec3 minP;
    glm::vec3 maxP;

    __host__ __device__ BBox() : minP(glm::vec3(FLT_MAX)), maxP(glm::vec3(FLT_MIN)) {}
    __host__ __device__ BBox(const glm::vec3& min, const glm::vec3& max)
    {
        minP = min;
        maxP = max;
    }
    __host__ __device__ BBox(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)
    {
        minP = glm::vec3(glm::min(p1.x, p2.x), glm::min(p1.y, p2.y), glm::min(p1.z, p2.z));
        maxP = glm::vec3(glm::max(p1.x, p2.x), glm::max(p1.y, p2.y), glm::max(p1.z, p2.z));

        minP = glm::vec3(glm::min(p3.x, minP.x), glm::min(p3.y, minP.y), glm::min(p3.z, minP.z));
        maxP = glm::vec3(glm::max(p3.x, maxP.x), glm::max(p3.y, maxP.y), glm::max(p3.z, maxP.z));
    }

    __host__ __device__ glm::vec3 getDiagonal() const
    {
        return maxP - minP;
    }

    __host__ __device__ int MaximumExtent() const
    {
        glm::vec3 diag = getDiagonal();
        if (diag.x > diag.y && diag.x > diag.z)
            return 0;
        else if (diag.y > diag.z)
            return 1;
        else
            return 2;
    }

};

static BBox Union(const BBox& bb1, const BBox& bb2)
{
    return BBox(glm::vec3(glm::min(bb1.minP.x, bb2.minP.x), glm::min(bb1.minP.y, bb2.minP.y), glm::min(bb1.minP.z, bb2.minP.z)),
        glm::vec3(glm::max(bb1.maxP.x, bb2.maxP.x), glm::max(bb1.maxP.y, bb2.maxP.y), glm::max(bb1.maxP.z, bb2.maxP.z)));
}

struct Triangle
{
    glm::vec3 verts[3];
    glm::vec3 nors[3];
    glm::vec2 uvs[3];

    BBox bbox = BBox();
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int triStart;
    int triEnd;

    BBox box;
};

struct Material {
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

struct Camera {
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
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

struct BVHPrimitiveInfo 
{
    int id;
    BBox bbox;
    glm::vec3 centroid;

    __host__ __device__ BVHPrimitiveInfo(int id, const BBox& bbox, const glm::vec3& centroid) :
        id(id),
        bbox(bbox),
        centroid(centroid)
    {}
};

struct BVHNode 
{
    BBox bbox;
    int left = -1;
    int right = -1;

    int splitAxis = -1;
    int firstPrimOffset = -1;
    int nPrimitives = -1;
    
};