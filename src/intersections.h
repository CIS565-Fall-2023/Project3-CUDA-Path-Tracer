#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

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

__host__ __device__ float triangleIntersectionTest(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, 
    glm::vec3 &bary) {
    glm::vec3 b;
    bool hit = glm::intersectRayTriangle(orig, dir, v1, v2, v3, bary);
    if (!hit) {
        hit = glm::intersectRayTriangle(orig, dir, v3, v2, v1, bary);
        if (!hit)return -1;
    }
    float t = bary.z;
    bary.z = 1 - bary.x - bary.y;
    return t;
}