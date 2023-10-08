#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH_PRIM
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

    //mesh only properties
    //mesh tri indices
    int tri_start_index{ -1 }, tri_end_index{ -1 };

    //bounding box min, max
    glm::vec3 bb_min, bb_max;
    
    glm::vec3 base_color{ glm::vec3(-1.f) }; //base color
    int texture_index{ -1 }, normal_map_index{ -1 };
};

// mesh pts include pos, tex color, normal(preprocessed on load)
struct MeshPoint {
    glm::vec3 pos;
    glm::vec3 normal; //normal as given from gltf data(not normal map)
    glm::vec2 tex_uv; //uv coords(assumed 1 set per mesh)
};

struct Triangle {
    MeshPoint points[3];
    int mesh_index{ -1 };
};

struct ImageInfo {
    int data_start_index;
    int img_w, img_h;
};

struct Material {
    //base color (used if no texture)
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
  glm::vec3 surfaceNormal;
  int materialId;

  //for mesh tex interp
  glm::vec3 bary;
  Geom* geom;
  Triangle* int_tri;
};

//BVH tree node
struct BVHNode {
    glm::vec3 min, max;
    int leftNode{ -1 };
    int triIndexStart, triCount{ 0 };
    int geomIndexStart, geomCount{ 0 };
};

//used to partition triangles in BVH construction and indexing
struct BVHTriIndex {
    int triIndex;
    glm::vec3 gFrameCentroid; //compute once in BVH construction
};

//used to partition geoms in BVH construction + indexing
struct BVHGeomIndex {
    int geomIndex;
    glm::vec3 gFrameCentroid; //compute once in BVH construction
};