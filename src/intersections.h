#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "bvh.h"

class Hittable {
public:
    virtual bool intersectRay(const Ray& ray, glm::vec3& intersectPt, glm::vec3& out_norm);
};

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
//__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
//        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
//    float radius = .5;
//
//    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
//    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
//
//    Ray rt;
//    rt.origin = ro;
//    rt.direction = rd;
//
//    float vDotDirection = glm::dot(rt.origin, rt.direction);
//    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
//    if (radicand < 0) {
//        return -1;
//    }
//
//    float squareRoot = sqrt(radicand);
//    float firstTerm = -vDotDirection;
//    float t1 = firstTerm + squareRoot;
//    float t2 = firstTerm - squareRoot;
//
//    float t = 0;
//    if (t1 < 0 && t2 < 0) {
//        return -1;
//    } else if (t1 > 0 && t2 > 0) {
//        t = min(t1, t2);
//        outside = true;
//    } else {
//        t = max(t1, t2);
//        outside = false;
//    }
//
//    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
//
//    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
//    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
//    if (!outside) {
//        normal = -normal;
//    }
//
//    return glm::length(r.origin - intersectionPoint);
//}

__host__ __device__ bool triangleIntersectionTest(
    const glm::vec3& orig,const glm::vec3& dir,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    glm::vec3& bary, float& t) {
    //bool result = glm::intersectRayTriangle(orig, dir, v0, v1, v2, bary);
    //t = bary.z;
    //bary.z = 1. - bary.x - bary.y;
    //return result;
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;

    glm::vec3 p = glm::cross(dir, e2);
    float det = glm::dot(e1, p);
    if (det < EPSILON && det > -EPSILON)return false;

    float inv_det = 1.0f / det;
    glm::vec3 s = orig - v0;
    bary.x = glm::dot(s, p) * inv_det;
    if (bary.x < 0.f || bary.x > 1.f)return false;

    glm::vec3 q = glm::cross(s, e1);
    bary.y = glm::dot(dir, q) * inv_det;
    if (bary.y < 0.f || (bary.x + bary.y) > 1.f)return false;

    bary.z = 1 - bary.x - bary.y;
    t = inv_det * glm::dot(e2, q);
    return  t > 0;
}

__host__ __device__ bool AABBIntersectionTest(
    const glm::vec3& orig
    , const glm::vec3& dir
    , const BoundingBox& b
    , float& t
) {

    glm::vec3 inv_dir = glm::vec3(1.f) / dir;
    glm::vec3 t_min = (b.minBound - orig) * inv_dir;
    glm::vec3 t_max = (b.maxBound - orig) * inv_dir;

    glm::vec3 t_near = glm::min(t_min, t_max);
    glm::vec3 t_far = glm::max(t_min, t_max);

    float t0 = glm::max(glm::max(t_near.x, t_near.y), t_near.z);
    float t1 = glm::min(glm::min(t_far.x, t_far.y), t_far.z);

    if (t0 > t1) return false;
    t = t0;
    return true;
}