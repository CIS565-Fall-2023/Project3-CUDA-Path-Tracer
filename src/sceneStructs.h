#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "bound.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
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
    struct {
        glm::vec3 v0;
        glm::vec3 v1;
        glm::vec3 v2;
    } triangle;

    Bound getBounds() const {
        glm::vec3 objectSpaceMin;
        glm::vec3 objectSpaceMax;

        // Get the object-space bounding box
        switch (type) {
        case TRIANGLE:
            objectSpaceMin = glm::min(triangle.v0, glm::min(triangle.v1, triangle.v2));
            objectSpaceMax = glm::max(triangle.v0, glm::max(triangle.v1, triangle.v2));
            break;
        case SPHERE:
            objectSpaceMin = glm::vec3(-scale.x);
            objectSpaceMax = glm::vec3(scale.x);
            break;
        case CUBE:
            objectSpaceMin = -0.5f * scale;
            objectSpaceMax = 0.5f * scale;
            break;
        default:
            objectSpaceMin = glm::vec3(0);
            objectSpaceMax = glm::vec3(0);
            break;
        }

        // Transform object-space bounding box corners to world space
        glm::vec3 corners[8] = {
            objectSpaceMin,
            glm::vec3(objectSpaceMax.x, objectSpaceMin.y, objectSpaceMin.z),
            glm::vec3(objectSpaceMin.x, objectSpaceMax.y, objectSpaceMin.z),
            glm::vec3(objectSpaceMin.x, objectSpaceMin.y, objectSpaceMax.z),
            glm::vec3(objectSpaceMax.x, objectSpaceMax.y, objectSpaceMin.z),
            glm::vec3(objectSpaceMax.x, objectSpaceMin.y, objectSpaceMax.z),
            glm::vec3(objectSpaceMin.x, objectSpaceMax.y, objectSpaceMax.z),
            objectSpaceMax
        };

        Bound bound;

        for (int i = 0; i < 8; i++) {
            glm::vec3 worldCorner = glm::vec3(transform * glm::vec4(corners[i], 1.0f));
            bound = bound.unionBound(worldCorner);
        }

        return bound;
    }
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