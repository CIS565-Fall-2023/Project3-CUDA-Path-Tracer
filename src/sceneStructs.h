#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    GLTF,
    OBJ,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    float surfaceArea() {
        glm::vec3 e = min - max;
        return 2.f * (e.x * e.y + e.x * e.z + e.y * e.z);
    };
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor;
};

struct Triangle
{
    int objectId;
    Vertex v0;
    Vertex v1;
    Vertex v2;
    glm::vec3 centroid;
    AABB aabb;

    void computeAABB() {
        aabb.min = glm::min(v0.pos, glm::min(v1.pos, v2.pos));
        aabb.max = glm::max(v0.pos, glm::max(v1.pos, v2.pos));
    }

    void computeCentroid() {
        centroid = (v0.pos + v1.pos + v2.pos) / 3.f;
    }
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::vec3 velocity;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int startTriIdx;
    int triangleCount;
    AABB aabb;
};

struct Material {
    glm::vec3 color;           // albedo
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float hasTransmission;
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
    glm::vec3 color;        // accumulated light
    int pixelIndex;
    int remainingBounces;
    glm::vec3 throughput;   // represents how materials we have encountered so far will alter the color of a light source when it scttaers off of them
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
