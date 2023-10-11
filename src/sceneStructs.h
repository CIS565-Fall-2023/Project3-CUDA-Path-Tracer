#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

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

struct Geom {
    enum GeomType type;
    int materialid;
    int meshid;
    int textureid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Texture {
    int texID;
    int width;
    int height;
    int start;
    int end;
};

struct Triangle {
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 v3;

    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 n3;

    glm::vec2 uv1;
    glm::vec2 uv2;
    glm::vec2 uv3;
};

struct GLTFMesh {
    //int materialid;
    glm::vec3 bbmin;
    glm::vec3 bbmax;
    int faceCount;
    std::vector<int> faceIndex;
    std::vector<Triangle> triangles;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
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
    glm::vec3 currentNormal;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 sourceNormal;
  glm::vec2 interUV;
  bool hasUV;
  int materialId;
  int textureId;
};


struct BVHNode {
    AABB bbox;
    int isLeaf;
    int leftIndex;
    int rightIndex;
    int parent;
    int triangleIndex;
};
