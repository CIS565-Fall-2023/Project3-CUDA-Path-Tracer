#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define BBOX_TRI_NUM 4
#define BVH_GPU_STACK_SIZE 64
#define BLOCK_SIZE_1D 64

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    int materialid;
    glm::vec3 pos[3];
    glm::vec2 uv[3];

    glm::vec3 min;
    glm::vec3 max;

    void set(){
        min = glm::vec3(
            glm::min(glm::min(pos[0].x, pos[1].x), pos[2].x),
            glm::min(glm::min(pos[0].y, pos[1].y), pos[2].y),
            glm::min(glm::min(pos[0].z, pos[1].z), pos[2].z)
        );
        max = glm::vec3(
            glm::max(glm::max(pos[0].x, pos[1].x), pos[2].x),
            glm::max(glm::max(pos[0].y, pos[1].y), pos[2].y),
            glm::max(glm::max(pos[0].z, pos[1].z), pos[2].z)
        );
    }
};

struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;
    int leftId;
    int rightId;
    int beginTriId;
    int triNum;

    BoundingBox() : min(), max(), leftId(-1), rightId(-1), beginTriId(-1), triNum(0) {}

    BoundingBox(glm::vec3 minPos, glm::vec3 maxPos) :
        min(minPos), max(maxPos), leftId(-1), rightId(-1), beginTriId(-1), triNum(0) {}

    BoundingBox(const Triangle& tri, int tai = -1) :
        min(tri.min), max(tri.max), leftId(-1), rightId(-1), beginTriId(-1), triNum(0) {}
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
    bool refractionBefore;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
