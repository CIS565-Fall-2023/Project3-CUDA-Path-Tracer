#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    TRI,
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
    bool no_intersection;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec3 intersectionPoint;
  int materialId;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor;
};

struct Triangle
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
    glm::vec3 centroid; // Dont forget to calculate this
};

struct Aabb {
    glm::vec3 bmin = glm::vec3(FLT_MAX);
    glm::vec3 bmax = glm::vec3(FLT_MIN);

    void grow(glm::vec3 input) {
		bmin = glm::min(bmin, input);
		bmax = glm::max(bmax, input);
	}

    void grow(const Triangle& tri) {
		grow(tri.v0.pos);
		grow(tri.v1.pos);
		grow(tri.v2.pos);
	}

    void grow(const Aabb &input) {
        grow(input.bmin);
        grow(input.bmax);
    }

    glm::vec3 extent() const {
        return bmax - bmin;
    }

    float surface_area() const
    {
        glm::vec3 e = extent() ;
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }

    float area()
    {
        glm::vec3 diff = bmax - bmin;
        return (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
    }

    Aabb()
        : bmin(glm::vec3(FLT_MAX)), bmax(glm::vec3(FLT_MIN))
		{}
};

struct BvhNode
{
    Aabb aa_bb;
    uint32_t left_first;
    uint32_t tri_count;

    BvhNode()
        : aa_bb(Aabb()), left_first(0), tri_count(0)
        {}

    BvhNode(uint32_t left_first, uint32_t tri_count)
        : aa_bb(), left_first(left_first), tri_count(tri_count)
        {}

    bool is_leaf()
    {
        return tri_count > 0;
    }
};

#define NUM_BINS 8

struct Bin
{
    Aabb bounds;
    int tri_count = 0;
};