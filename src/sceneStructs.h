#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "bound.h"
#include "geom.h"

#define BACKGROUND_COLOR (glm::vec3(0.5f))

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    float tMax = 1e38f;
};

struct BVHGeomInfo {
    size_t geomIndex;
    Bound bounds;
    glm::vec3 centroid;

    BVHGeomInfo(size_t geomIndex, const Bound& bounds)
        : geomIndex(geomIndex), bounds(bounds),
          centroid(.5f * bounds.pMin + .5f * bounds.pMax) {};
};

struct BVHNode {
    Bound bounds;
    BVHNode* left, * right;      
    int splitAxis, geomIndex, geomCount;
    
    void initLeaf(int first, int n, const Bound& bound) {
        geomIndex = first;
        geomCount = n;
        bounds = bound;
        left = right = nullptr;
    }

    void initInterior(int axis, BVHNode* leftChild, BVHNode* rightChild) {
        left = leftChild;
        right = rightChild;
        bounds = leftChild->bounds.unionBound(rightChild->bounds);
        splitAxis = axis;
        geomCount = 0;
    }
};

struct LinearBVHNode {
    Bound bounds;
    union {
        int geomIndex;        // offset to the geometry for leaf nodes;
        int rightChildOffset; // offset to the right child 
    };
    uint16_t geomCount;    // 0 -> interior node
    uint8_t axis;          // interior node: xyz
    uint8_t pad[1];        // ensure 32 byte total size
};

struct Material {
    glm::vec3 ambient = glm::vec3(0.1f);      // Debug, default to a low gray value
    glm::vec3 diffuse = glm::vec3(0.8f);      // Default to a neutral gray
    glm::vec3 transmittance = glm::vec3(0.5f); // Default to opaque, absortion
    struct {
        float exponent = 10.0f;              // Default shininess value
        glm::vec3 color = glm::vec3(1.0f);   // Default to white
    } specular;
    float roughness = 0.0f;                  // Midway between smooth and rough
    float metallic = 0.0f;                   // Default to non-metal
    float sheen = 0.0f;                      // Default to no sheen effect
    float hasReflective = 0.0f;              // Default to no reflection
    float hasRefractive = 0.0f;              // Default to no refraction
    float indexOfRefraction = 1.0f;          // Default to vacuum's IOR
    float emittance = 0.0f;                  // Default to no emission
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
    float aperture;
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