#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))


enum BsdfSampleType
{
    diffuse_refl = 1,
    spec_refl = 1 << 1,
    spec_trans = 1 << 2,
    spec_glass = 1 << 3,
    microfacet_refl = 1 << 4,
    plastic = 1 << 5,
    diffuse_trans = 1 << 6,
    microfacet_trans = 1 << 7
};


enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

__device__ static Ray SpawnRay(glm::vec3 pos, glm::vec3 wi) {
    return Ray{ pos + wi * 0.0001f, wi };
}

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
    glm::vec3 color = glm::vec3(0.0);
    struct {
        float exponent = 0.0;
        glm::vec3 color = glm::vec3(0.0);
    } specular;
    float hasReflective = 0.0;
    float hasRefractive = 0.0;
    float indexOfRefraction = 0.0;
    float emittance = 0.0;
    float roughness = 0.0;
    float eta = 0.0;
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
    glm::vec3 pos;
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    glm::vec3 woW;
};
