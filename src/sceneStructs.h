#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

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

    void Geom::getBounds(glm::vec3& minBounds, glm::vec3& maxBounds) const {
        switch (type) {
            case TRIANGLE:
                minBounds = glm::min(triangle.v0, glm::min(triangle.v1, triangle.v2));
                maxBounds = glm::max(triangle.v0, glm::max(triangle.v1, triangle.v2));
                break;

            default:
                minBounds = glm::vec3(0);
                maxBounds = glm::vec3(0);
                break;
            }
    }
};

struct BVHNode {
    glm::vec3 minBounds; // minimun bounding box corner
    glm::vec3 maxBounds; // maximun bounding box corner
    BVHNode* left;       // left child 
    BVHNode* right;      // right child
    int geomIndex;       // index of the geom for leaf node
    bool isLeafNode;     // flag for leaf node 
    
    BVHNode() :
        left(nullptr), right(nullptr), geomIndex(-1), isLeafNode(false) {}
};

struct CompactBVH {
    glm::vec3 minBounds;
    glm::vec3 maxBounds;
    union {
        int geomIndex;        // offset to the geometry for leaf nodes;
        int rightChildOffset; // offset to the right child 
    };
    uint16_t geomCount;       // number of geometries for leaf node
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