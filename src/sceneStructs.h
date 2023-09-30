#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

enum LightType {
    NONE = -1,
    AREA = 0,
    POINT = 1,
    SPOT = 2,
    ENVIRONMENT = 3
};

enum MaterialType {
    DIFFUSE = 0,
    SPECULAR_REFL = 1,
    SPECULAR_TRANS = 2,
    SPECULAR_FRES = 3,
    MICROFACET = 4,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 position[3];
    glm::vec3 normal[3];
    glm::vec2 texcoord[3];
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

    // Mesh, triangle start, end
    int triangleStart;
    int triangleEnd;

    // Bounding box
    glm::vec3 minPoint;
    glm::vec3 maxPoint;
};

struct Light {
    Geom geom;
    enum LightType lightType;
    float innerAngle, outerAngle; // for spot light
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

    MaterialType type;
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

    // Depth of field
    float lensRadius;
	float focalDistance;
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

    glm::vec3 throughput;

    bool isSpecularBounce;
    bool isFromCamera;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 surfaceTangent;
  int geomId;
  int materialId;
};
