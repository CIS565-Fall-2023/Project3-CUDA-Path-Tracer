#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
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
};

struct Geom {
    enum GeomType type;
    int geomId;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int startTriIdx;
    int endTriIdx;
    bool hasUVs = false;
    bool hasNormals = false;
};

struct AABB {
    glm::vec3 minPos;
    glm::vec3 maxPos;
    glm::vec3 centroid;
    int triIdx;
    Geom geom;
    AABB() : minPos(), maxPos(), centroid(), geom(), triIdx(-1) {}
    AABB(glm::vec3 minP, glm::vec3 maxP) : minPos(minP), maxPos(maxP), centroid(), geom(), triIdx(-1) {}
    AABB(glm::vec3 minP, glm::vec3 maxP, glm::vec3 c, Geom g, int id) : minPos(minP), maxPos(maxP), centroid(c), geom(g), triIdx(id) {}
    AABB(glm::vec3 minP, glm::vec3 maxP, glm::vec3 c, Geom g) : AABB(minP, maxP, c, g, -1) {}
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
#if DEPTH_OF_FIELD
    float lensRadius;
    float focalLength;
#endif
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
    glm::vec3 accumCol;
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
