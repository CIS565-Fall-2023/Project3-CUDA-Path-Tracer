#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct Image {
    std::vector<float> imgdata;
    int width;
    int height;
    int channels;
};

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
    int index;
    int textureIdx;
    enum GeomType type;
    int materialid;
    int triCount;
    int triIdx;
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

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
};

struct Triangle {
    Vertex vertices[3];
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
    glm::vec3 accumRay;
    bool dead;
    int pixelIndex;
    int remainingBounces;
    bool spec;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int hitGeomIdx;
  glm::vec2 uv;
};
