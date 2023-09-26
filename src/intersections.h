#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

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
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) 
{
    Ray local_ray(multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f)),
                   glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f))));

    glm::vec3 inv_dir = 1.f / local_ray.direction;
    glm::vec3 t_near = (glm::vec3(-0.5f) - local_ray.origin) * inv_dir;
    glm::vec3 t_far = (glm::vec3(0.5f) - local_ray.origin) * inv_dir;

    glm::vec3 t_min = glm::min(t_near, t_far);
    glm::vec3 t_max = glm::max(t_near, t_far);

    float t0 = glm::max(glm::max(t_min.x, t_min.y), t_min.z);
    float t1 = glm::min(glm::min(t_max.x, t_max.y), t_max.z);

    if (t0 > t1) return -1.f;

    if (t0 > 0.f)
    {
        glm::vec3 v1(t_min.y, t_min.z, t_min.x);
        glm::vec3 v2(t_min.z, t_min.x, t_min.y);

        normal = -glm::sign(local_ray.direction) * glm::step(v1, t_min) * glm::step(v2, t_min);
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(normal, 0.f)));
        intersectionPoint = r * t0;
        return t0;
    }
    if (t1 > 0.f)
    {
        glm::vec3 v1(t_max.y, t_max.z, t_max.x);
        glm::vec3 v2(t_max.z, t_max.x, t_max.y);

        normal = -glm::sign(local_ray.direction) * glm::step(v1, t_max) * glm::step(v2, t_max);
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(normal, 0.f)));
        intersectionPoint = r * t1;
        return t1;
    }
    return -1.f;
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
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
