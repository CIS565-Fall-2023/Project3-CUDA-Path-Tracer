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
    PROCEDURAL
};

enum class ProceduralType : uint8_t
{
    Plane,
    Cube,
    Sphere,
    Cylinder,
    Capsule,
    Torus,
    opSmoothUnion,
    opSmoothSubtraction,
    opSmoothIntersection
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
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
    Triangle triangle;
    ProceduralType proceduralType;
};

enum class MaterialType : uint8_t
{
    None,
    Emissive,
    Diffuse,
    Metal,
    Glass,
    Image
};

enum class Pattern : uint8_t
{
    None,
    Ring,
    CheckerBoard,
    PerlinNoise
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular{};
    float hasReflective = 0.0f;
    float hasRefractive = 0.0f;
    float indexOfRefraction = 1.0f;
    float emittance = 0.0f;
    float fuzz = 0.1f;
    MaterialType type = MaterialType::None;
    Pattern pattern = Pattern::None;;
    cudaTextureObject_t albedo = 0;
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
    bool needSkyboxColor;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId = -1;
  bool frontFace;
  glm::vec3 point;
  float u;
  float v;
};

struct SortIntersection
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
	{
		return a.materialId < b.materialId;
	}
};

struct returnRemainBounce
{
    __host__ __device__
        bool operator()(const PathSegment& a)
    {
        return a.remainingBounces > 0;
    }
};


