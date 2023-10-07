#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    GLTF_MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};


struct Triangle
{
    std::vector<glm::vec3> pos{ std::vector<glm::vec3>() };
};

struct SceneMesh
{
    const unsigned short* indices;
    bool hasIndices{ false };
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    bool hasNormals{ false };

    int startTriIdx{ -1 };
    int endTriIdx{ -1 };

    SceneMesh() :
        indices(), hasIndices(false), positions(std::vector<glm::vec3>()), normals(std::vector<glm::vec3>())
    {}
};

/// <summary>
/// Used for the case where there's multiple meshes in a GLTF scene
/// </summary>
struct SceneMeshGroup
{
    bool valid{ false };
    int startTriIdx{ -1 };
    int endTriIdx{ -1 };
    int startMeshIdx{ -1 };
    int endMeshIdx{ -1 };
};

struct Geom {
    enum GeomType type;
    SceneMeshGroup meshGroup;
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

    // Thin lens camera
    float apertureSize;
    float focalLength;
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
    glm::vec3 accum_throughput;     // throughput, we'll only use it if we eventually hit a light source
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};