#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "bvh.h"
/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ inline float util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, glm::vec3* normal = nullptr)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = r.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (pMin[xyz] - r.origin[xyz]) / qdxyz;
            float t2 = (pMax[xyz] - r.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax > tmin && tmax > 0) {
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
        }
        if (normal)
            (*normal) = glm::normalize(tmin_n);
        return tmin;
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(const Object& box,const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal) {
    Ray q;
    q.origin    =                multiplyMV(box.Transform.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 local_n;
    float t = util_geometry_ray_box_intersection(glm::vec3(-0.5f), glm::vec3(0.5f), q, &local_n);

    if (t>0) {
        intersectionPoint = multiplyMV(box.Transform.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        normal = glm::normalize(multiplyMV(box.Transform.invTranspose, glm::vec4(local_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}


__host__ __device__ bool boundingBoxIntersectionTest(const BoundingBox& bbox, const Ray& r)
{
    return util_geometry_ray_box_intersection(bbox.pMin, bbox.pMax, r) > 0.0;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(const Object& sphere,const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
    } else {
        t = max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.Transform.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.Transform.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool util_geometry_ray_triangle_intersection(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& t)
{
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 N = glm::cross(v0v1, v0v2); // N

    float NdotRayDir = glm::dot(N, dir);
    if (abs(NdotRayDir) < 1e-10f) 
        return false; 

    float d = glm::dot(-N, v0);

    t = -(glm::dot(N, orig) + d) / NdotRayDir;

    if (t < 0) return false; 
    glm::vec3 P = orig + t * dir;

    glm::vec3 C; 

    // edge 0
    glm::vec3 edge0 = v1 - v0;
    glm::vec3 vp0 = P - v0;
    C = glm::cross(edge0, vp0);
    if (glm::dot(N, C) < 0) return false; 

    // edge 1
    glm::vec3 edge1 = v2 - v1;
    glm::vec3 vp1 = P - v1;
    C = glm::cross(edge1,vp1);
    if (glm::dot(N,C) < 0)  return false; 

    // edge 2
    glm::vec3 edge2 = v0 - v2;
    glm::vec3 vp2 = P - v2;
    C = glm::cross(edge2, vp2);
    if (glm::dot(N,C) < 0) return false;

    return true; 
}

__host__ __device__ float triangleIntersectionTest(const ObjectTransform& Transform, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& baryCoord)
{
    float t = -1.0;
    glm::vec3 v0w = multiplyMV(Transform.transform, glm::vec4(v0, 1.0f));
    glm::vec3 v1w = multiplyMV(Transform.transform, glm::vec4(v1, 1.0f));
    glm::vec3 v2w = multiplyMV(Transform.transform, glm::vec4(v2, 1.0f));
    if (util_geometry_ray_triangle_intersection(r.origin, r.direction, v0w, v1w, v2w, t))
    {
        normal = glm::cross(glm::vec3(v1w - v0w), glm::vec3(v2w - v0w));
        normal = glm::normalize(normal);
        intersectionPoint = r.origin + r.direction * t;
        return t;
    }
    else
    {
        return -1;
    }
}
