#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    LIGHT,
    OBJMESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 direction_inv;
};

struct Triangle
{
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
    glm::vec3 plane_normal;
    float S;
    int mat_ID;
};

struct TriBounds {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    glm::vec3 AABB_centroid;
    int tri_ID;
};

struct BVHNode {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    BVHNode* child_nodes[2];
    int split_axis;
    int tri_index;
};

struct BVHNode_GPU {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    int tri_index;
    int offset_to_second_child;
    int axis;
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
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
    int triangleIdStart;
    int triangleIdEnd;
    int textureId = -1;
    int bumpTextureId = -1;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float hasSubsurface;
    float subsurfaceRadius;
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

    float aperture = 0.;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct Texture {
    int id;
    int channel;
    int width;
    int height;
    int idx;
    float* bumpMap = nullptr;
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
  glm::vec2 uv;
  int textureId = -1;
  int bumpTextureId = -1;
};
