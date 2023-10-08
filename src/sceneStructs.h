#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE
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
};

struct Mesh {
    int materialid;
    float* vertices;
    unsigned short* indices;
    int numVertices;
    int numIndices;
    Geom boundingVolume;
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Triangle {
    glm::vec3 vertices[3];
    glm::vec3 centroid;
};

static const int OCTREE_MAX_DEPTH = 1;
static const int OCTREE_MAX_PRIMITIVES = 10;
static const float OCTREE_MIN_BOX_SIZE = 0.5f;
struct OctreeNode {
    int children[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
};

struct Octree {
    int root;
    std::vector<OctreeNode> nodes;
    std::vector<Triangle> triangles;
    std::vector<Geom> boundingBoxes;
    std::vector<int> dataStarts;
};

struct OctreeDev {
    int root;
    OctreeNode* nodes;
    Triangle* triangles;
    Geom* boundingBoxes;
    int* dataStarts;
    int numNodes;
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
  glm::vec3 intersectPoint;
  glm::vec3 surfaceNormal;
  int materialId;
};
