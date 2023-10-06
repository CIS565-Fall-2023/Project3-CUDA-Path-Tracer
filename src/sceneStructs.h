#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"
#define BACKGROUND_COLOR (glm::vec3(0.0f))

struct TriangleDetail;
struct TBB {
    TBB();
    TBB(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2);
    TBB(glm::vec3 min, glm::vec3 max);
    glm::vec3 min, max;
    float area() const;
    inline void expand(const glm::vec3& p) {
        this->max = glm::max(this->max, p);
        this->min = glm::min(this->min, p);
    }
    inline void expand(const TBB& other)
    {
        expand(other.max);
        expand(other.min);
    }
};


struct TBVHNode {
    TBB tbb;
    bool isLeaf;
    int left, right;
    int triId = -1;
    int miss, base;
};

class TBVH
{
public:
    TBVH() = default;
    TBVH(std::vector<TriangleDetail>& tris, TBB& tbb);
    std::vector<std::vector<TBVHNode>> nodes;
    int nodesNum = -1;
private:
    int splitBVH(std::vector<TriangleDetail>& triangles, std::vector<int> objIdx, int num, TBB& tbb, int face);
    void reorderNodes(std::vector<TriangleDetail>& triangles, int face, int index);
    int reorderTree(std::vector<TriangleDetail>& triangles, int face, int index);
    void setLeftMiss(int id, int idParent, int face);
    void setRightMiss(int id, int idParent, int face);
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

__device__ static Ray SpawnRay(glm::vec3 pos, glm::vec3 wi) {
    return Ray{ pos + wi * 0.0001f, wi };
}

struct Transformation {
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct TriangleDetail {
    TriangleDetail(Transformation t, int materialid, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
        glm::vec3 normal0, glm::vec3 normal1, glm::vec3 normal2,
        glm::vec4 tangent0, glm::vec4 tangent1, glm::vec4 tangent2,
        glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2, bool doubleSided, int id)
        : t(t), materialid(materialid), v0(v0), v1(v1), v2(v2),
        normal0(normal0), normal1(normal1), normal2(normal2),
        tangent0(tangent0), tangent1(tangent1), tangent2(tangent2),
        uv0(uv0), uv1(uv1), uv2(uv2), centroid((v0 + v1 + v2)* (1.f / 3.f)),
        tbb(glm::vec3(t.transform* glm::vec4(v0, 1.f)), glm::vec3(t.transform* glm::vec4(v1, 1.f)), glm::vec3(t.transform* glm::vec4(v2, 1.f))), doubleSided(doubleSided), id(id) {}
    Transformation t;
    int materialid;
    glm::vec3 v0, v1, v2;
    glm::vec3 normal0, normal1, normal2;
    glm::vec4 tangent0, tangent1, tangent2;
    glm::vec3 centroid;
    glm::vec2 uv0, uv1, uv2;
    bool doubleSided;
    TBB tbb;
    int id = -1;
};

struct TextureInfo {
    int index{ -1 };
    int width;
    int height;
    int channel;
    cudaTextureObject_t cudaTexObj;
    size_t size;
};

struct NormalTextureInfo {
    int index{ -1 };    // required
    int texCoord{ 0 };  // The set index of texture's TEXCOORD attribute used for
    // texture coordinate mapping.
    double scale{
        1.0 };  // scaledNormal = normalize((<sampled normal texture value>
    // * 2.0 - 1.0) * vec3(<normal scale>, <normal scale>, 1.0))
    cudaTextureObject_t cudaTexObj;

    NormalTextureInfo() = default;
    NormalTextureInfo(int _index, int _texCoord, double _scale, cudaTextureObject_t _cudaTexObj) :
        index(_index), texCoord(_texCoord), scale(_scale), cudaTexObj(_cudaTexObj) {}
    DEFAULT_METHODS(NormalTextureInfo)
        bool operator==(const NormalTextureInfo&) const;
};


struct OcclusionTextureInfo {
    int index{ -1 };    // required
    int texCoord{ 0 };  // The set index of texture's TEXCOORD attribute used for
    // texture coordinate mapping.
    double strength{ 1.0 };  // occludedColor = lerp(color, color * <sampled
    // occlusion texture value>, <occlusion strength>)
    cudaTextureObject_t cudaTexObj;

    // Filled when SetStoreOriginalJSONForExtrasAndExtensions is enabled.

    OcclusionTextureInfo() = default;
    OcclusionTextureInfo(int _index, int _texCoord, double _strength, cudaTextureObject_t _cudaTexObj) :
        index(_index), texCoord(_texCoord), strength(_strength), cudaTexObj(_cudaTexObj) {}
    DEFAULT_METHODS(OcclusionTextureInfo)
        bool operator==(const OcclusionTextureInfo&) const;
};

struct PbrMetallicRoughness {
    glm::vec4 baseColorFactor;  // len = 4. default [1,1,1,1]
    TextureInfo baseColorTexture;
    double metallicFactor{ 1.0 };   // default 1
    double roughnessFactor{ 1.0 };  // default 1
    TextureInfo metallicRoughnessTexture;

    PbrMetallicRoughness()
        : baseColorFactor(glm::vec4{ 1. }) {}
    DEFAULT_METHODS(PbrMetallicRoughness)
        bool operator==(const PbrMetallicRoughness&) const;
};

enum BsdfSampleType
{
    DIFFUSE_REFL = 1 << 1,
    SPEC_REFL = 1 << 2,
    SPEC_TRANS = 1 << 3,
    MICROFACET_REFL = 1 << 4,
    MICROFACET_TRANS = 1 << 5,
    PLASTIC = 1 << 6,
    DIFFUSE_TRANS = 1 << 7
};

struct Material {
    enum Type {
        UNKNOWN = 0,
        DIFFUSE = BsdfSampleType::DIFFUSE_REFL,
        DIELECTRIC = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS,
        SPECULAR = BsdfSampleType::SPEC_REFL,
        METAL = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS | 1,
        ROUGH_DIELECTRIC = BsdfSampleType::MICROFACET_REFL | BsdfSampleType::MICROFACET_TRANS,
        PBR = DIFFUSE | DIELECTRIC,
        LIGHT = 1 << 7
    };
    uint32_t type = Type::DIFFUSE;

    glm::vec3 emissiveFactor = glm::vec3(1.f);  // length 3. default [0, 0, 0]
    float emissiveStrength = 1.f;
    // std::string alphaMode;               // default "OPAQUE"
    double alphaCutoff{ 0.5 };             // default 0.5
    bool doubleSided{ false };             // default false;

    PbrMetallicRoughness pbrMetallicRoughness;

    NormalTextureInfo normalTexture;
    OcclusionTextureInfo occlusionTexture;
    TextureInfo emissiveTexture;

    struct Dielectric {
        float eta = 1.55f;
    }dielectric;

    struct Specular {
        float specularFactor = 0.f;
        glm::vec3 specularColorFactor = glm::vec3(0.f);
    }specular;

    struct Metal {
        glm::vec3 etat = glm::vec3(0.f);
        glm::vec3 k = glm::vec3(0.f);
    }metal;

    __host__ __device__ Material() : dielectric(), metal() {}
    __host__ __device__ ~Material() = default;
    __host__ __device__ Material(const Material&) = default;
    __host__ __device__ Material(Material&&) TINYGLTF_NOEXCEPT = default;
    __host__ __device__ Material& operator=(const Material&) = default;
    __host__ __device__ Material& operator=(Material&&) TINYGLTF_NOEXCEPT = default;
    __host__ __device__ bool operator==(const Material&) const;
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
    float focalLength;
    float apertureSize;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    bool isCached = false;
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
    glm::vec2 uv;
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
    glm::vec2 uv;
    glm::vec4 tangent;
};
