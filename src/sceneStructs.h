#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MODEL
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

struct Geom {
    enum GeomType type;
    int materialid;
    ObjectTransform Transform;
};

struct Model {
    int materialid;
    int triangleStart, triangleEnd;
    ObjectTransform Transform;
};

enum MaterialType {
    diffuse = 0x1, frenselSpecular = 0x2, emitting = 0x4
};

struct Material {
    glm::vec3 color;
    float indexOfRefraction;
    float emittance;
    int type;
    Material():color(glm::vec3(0)), indexOfRefraction(0), emittance(0), type(1) {}//default diffuse
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
  glm::vec3 worldPos;
  int materialId;
};
