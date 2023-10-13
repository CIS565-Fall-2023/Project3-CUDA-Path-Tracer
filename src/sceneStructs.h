#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    int bvhRootIdx;
    int primStartIdx;
    int primCnt;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
};

struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
    glm::vec3 centroid;
    int materialid;
};

struct AABoundBox {
    glm::vec3 minCoord = glm::vec3(FLT_MAX);
    glm::vec3 maxCoord = glm::vec3(-FLT_MAX);
    __host__ __device__ void grow(const glm::vec3& p) {
        minCoord = min(minCoord, p);
        maxCoord = max(maxCoord, p);
    }
    __host__ __device__ void grow(const Triangle& t) {
        grow(t.v1.pos);
        grow(t.v2.pos);
        grow(t.v3.pos);
    }
    __host__ __device__ void grow(const AABoundBox& o) {
        grow(o.minCoord);
        grow(o.maxCoord);
    }
    __host__ __device__ float surfaceArea() const {
        glm::vec3 diag = maxCoord - minCoord;
        return diag.x * diag.y + diag.y * diag.z + diag.z * diag.x;
    }
};

struct BVHNode {
    AABoundBox bounds;
    int leftInd; // first index of triangle if leaf, else index of left child node
    int primCnt; // only non-zero if it's a leaf (only leaf has primitives)
    
    BVHNode() : bounds(AABoundBox()), leftInd(0), primCnt(0) {}
    BVHNode(int leftInd, int primCnt) : leftInd(leftInd), primCnt(primCnt) {}
    BVHNode(AABoundBox bounds, int leftInd, int primCnt) : bounds(bounds), leftInd(leftInd), primCnt(primCnt) {}

    __host__ __device__ bool isLeaf() const {
        return primCnt > 0;
    }
    __host__ __device__ float scanCost() const {
        return primCnt * bounds.surfaceArea();
    }
};

struct BVHBin {
    AABoundBox bounds;
    int primCnt = 0;
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
    int textureid = -1;
};

struct Texture {
    float* host_buffer;
    int width;
    int height;
    int channels;
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
    float focalDistance;
    float lensRadius;
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
  glm::vec2 uv;
  int materialId;

  __host__ __device__
  bool operator<(const ShadeableIntersection& o) const {
      return materialId < o.materialId;
  }
};
