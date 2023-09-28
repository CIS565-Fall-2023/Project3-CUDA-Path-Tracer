#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "common.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray 
{
    glm::vec3 origin;
    glm::vec3 direction;
    CPU_GPU Ray(const glm::vec3& o = {0, 0, 0}, const glm::vec3& d = { 0, 0, 0 })
        :origin(o), direction(d)
    {}

    CPU_GPU glm::vec3 operator*(const float& t) const { return origin + t * direction; }

public:
    CPU_GPU static Ray SpawnRay(const glm::vec3& o, const glm::vec3& dir)
    {
        return { o + dir * 0.001f, dir };
    }
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
    glm::vec3 albedo{0.f};
    float emittance = 0.f;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 ref;
    glm::vec3 forward;
    glm::vec3 up;
    glm::vec3 right;
    float fovy;
    glm::vec2 pixelLength;

    CPU_ONLY void Recompute() 
    {
        forward = glm::normalize(ref - position);
        right = glm::normalize(glm::cross(forward, {0, 1, 0}));
        up = glm::normalize(glm::cross(right, forward));
    }
    CPU_GPU Ray CastRay(const glm::vec2& p)
    {
        glm::vec2 ndc = 2.f * p / glm::vec2(resolution); 
        ndc.x = 1.f - ndc.x;
        ndc.y = 1.f - ndc.y;

        float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);

        // point in camera space
        float radian = glm::radians(fovy * 0.5f);
        glm::vec3 p_camera = glm::vec3(
            ndc.x * glm::tan(radian) * aspect,
            ndc.y * glm::tan(radian),
            1.f
        );

        Ray ray(glm::vec3(0), p_camera);

        // transform to world space
        ray.origin = position + ray.origin.x * right + ray.origin.y * up;
        ray.direction = glm::normalize(
            ray.direction.z * forward +
            ray.direction.y * up +
            ray.direction.x * right
        );

        return ray;
    }
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::vector<uchar4> c_image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput{ 1, 1, 1 };
    glm::vec3 radiance{0, 0, 0};
    int pixelIndex;
    int remainingBounces;
    CPU_GPU void Reset() 
    {
        throughput = glm::vec3(1.f);
        radiance = glm::vec3(0.f);
        pixelIndex = 0;
    }
    CPU_GPU void Terminate() { remainingBounces = 0; }
    CPU_GPU bool IsEnd() const { return remainingBounces <= 0; }
};

struct Intersection
{
    int shapeId;
    int materialId;
    float t;
    glm::vec2 uv; // local uv
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection 
{
    float t;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    int materialId;

    CPU_GPU void Reset()
    {
        t = -1.f;
        materialId = -1.f;
    }
};

struct AABB
{
    CPU_ONLY AABB(const glm::vec3& _min = glm::vec3(Float_MAX),
                  const glm::vec3& _max = glm::vec3(Float_MIN))
        : m_Min(_min), m_Max(_max)
    {}
    glm::vec3 m_Min;
    glm::vec3 m_Max;
    
    glm::ivec3 m_Data; // leaf data or node data

    inline CPU_ONLY void Merge(const AABB& other)
    {
        m_Min = glm::min(m_Min, other.m_Min);
        m_Max = glm::max(m_Max, other.m_Max);
    }
    inline CPU_ONLY void Merge(const glm::vec3& p)
    {
        m_Min = glm::min(p, m_Min);
        m_Max = glm::max(p, m_Max);
    }
    inline CPU_ONLY glm::vec3 GetDiagnol() const { return m_Max - m_Min; }
    inline CPU_ONLY glm::vec3 GetCenter() const { return glm::vec3(0.5f) * (m_Min + m_Max);  }
    inline CPU_ONLY int GetMaxAxis() const
    {
        glm::vec3 d = GetDiagnol();
        return ((d.x > d.y && d.x > d.z) ? 0 : ((d.y > d.z) ? 1 : 2));
    }
    inline CPU_ONLY float GetCost() const
    {
        glm::vec3 d = GetDiagnol();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    inline CPU_GPU bool Intersection(const Ray& ray, const glm::vec3& inv_dir, float& t) const 
    {
        glm::vec3 t_near = (m_Min - ray.origin) * inv_dir;
        glm::vec3 t_far = (m_Max - ray.origin) * inv_dir;

        glm::vec3 t_min = glm::min(t_near, t_far);
        glm::vec3 t_max = glm::max(t_near, t_far);

        t = glm::max(glm::max(t_min.x, t_min.y), t_min.z);

        if (t > glm::min(glm::min(t_max.x, t_max.y), t_max.z)) return false;

        return true;
    }
};

struct TriangleIdx
{
    TriangleIdx(const glm::ivec3 v, 
                const glm::ivec3& n, 
                const glm::ivec3& uv, 
                const unsigned int& material)
        :v_id(v), n_id(n), uv_id(uv), material(material)
    {}
    glm::ivec3 v_id;
    unsigned int material;
    glm::ivec3 n_id;
    glm::ivec3 uv_id;
};