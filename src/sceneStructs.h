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
    SPEC_REFL = 1,
    SPEC_TRANS = 2,
    SPEC_FRESNEL = 3,
    MICROFACET = 4
};

enum MediumType {
    NONSCATTERING = 0,
    ISOTROPIC = 1,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 position[3];
    glm::vec3 normal[3];
    //glm::vec2 texcoord[3];
};

// Bounding box
struct AABB {
    glm::vec3 minPoint;
    glm::vec3 maxPoint;
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

    AABB aabb;

    // if type == MESH
    Triangle triangle;
};

struct Light {
    Geom geom;
    enum LightType lightType;
    float innerAngle, outerAngle; // for spot light
};

struct Medium {
    bool valid;

    enum MediumType mediumType;
    glm::vec3 absorptionCoefficient; // -log(absorptionColor) / absorptionAtDistance
    float scatteringCoefficient; // for isotropic scattering medium
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;

    float reflectivity;
    float refractivity;
    float roughness;

    float indexOfRefraction; // refraction index, eta
    float emittance; // light

    MaterialType type;
    // texture maps
    int albedoTex;
    int normalTex;
    int roughnessTex;

    // medium
    Medium medium;
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

    // near and far plane
    float farClip = 1000.f;
    float nearClip = 0.001f;
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
    int russianRouletteThres;

    glm::vec3 throughput;

    bool isSpecularBounce;
    bool isFromCamera;

    Medium medium; // check current medium, null means vaccum
    bool hitSurface; // check if hit surface
    float tFar;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec3 surfaceTangent;

    float IORi = 1.0f;
    float IORo;

    int geomId;
    int materialId;
};

// build in CPU
struct KDNode {
    KDNode* leftChild;
    KDNode* rightChild;

    unsigned int axis;
    std::vector<Geom> geoms;

	AABB aabb;

    KDNode() : leftChild(nullptr), rightChild(nullptr), axis(0) {
        aabb.maxPoint = glm::vec3(FLT_MIN);
        aabb.minPoint = glm::vec3(FLT_MAX);
    }
};

struct KDAccelNode{
    int geomStart;
    int numGeoms;

    // leftOffset = id + 1
    int rightOffset;
    int axis;
    AABB aabb;
};