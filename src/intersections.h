#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

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
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
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

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
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

__host__ __device__ float triangleIntersectionTest(Geom custom_obj, Ray r,
    glm::vec3& intersectionPoint, Triangle* triangles, int triangles_start, int triangles_end, glm::vec3& normal, bool& outside,
    glm::vec2& uv)
{

    // get the Ray in local space
    Ray ray_inversed;
    ray_inversed.origin = multiplyMV(custom_obj.inverseTransform, glm::vec4(r.origin, 1.0f));
    ray_inversed.direction = glm::normalize(multiplyMV(custom_obj.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float min_t = FLT_MAX;

    for (int i = triangles_start; i < triangles_end; i++)
    {
        Triangle& triangle = triangles[i];
        glm::vec3 vertices[3];
        glm::vec3 normals[3];
        glm::vec2 uvs[3];

        for (int j = 0; j < 3; j++) {
            vertices[j] = triangle.vertices[j];
            normals[j] = triangle.normals[j];
            uvs[j] = triangle.uvs[j];
        }
        glm::vec3 baryPos;


        // Not intersected
        if (glm::intersectRayTriangle(ray_inversed.origin, ray_inversed.direction, vertices[0], vertices[1], vertices[3], baryPos))
        {

            // Smooth interpolate normals
            glm::vec3 n0;
            glm::vec3 n1;
            glm::vec3 n2;

            glm::vec3 isect_pos = (1.f - baryPos.x - baryPos.y) * vertices[1] + baryPos.x * vertices[2] + baryPos.y * vertices[3];
            intersectionPoint = multiplyMV(custom_obj.transform, glm::vec4(isect_pos, 1.f));
            float t = glm::length(r.origin - intersectionPoint);
            if (t > min_t)
            {
                continue;
            }
            min_t = t;

            if ((glm::length(normals[0]) != 0) && (glm::length(normals[1]) != 0) && (glm::length(normals[2]) != 0))
            {
                n0 = normals[0];
                n1 = normals[1];
                n2 = normals[2];
            }
            else
            {
                n0 = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
                n1 = glm::normalize(glm::cross(vertices[0] - vertices[1], vertices[2] - vertices[1]));
                n2 = glm::normalize(glm::cross(vertices[0] - vertices[2], vertices[1] - vertices[2]));
            }

            // Barycentric Interpolation
            float S = 0.5f * glm::length(glm::cross(vertices[0] - vertices[1], vertices[2] - vertices[1]));
            float S0 = 0.5f * glm::length(glm::cross(vertices[1] - isect_pos, vertices[2] - isect_pos));
            float S1 = 0.5f * glm::length(glm::cross(vertices[0] - isect_pos, vertices[2] - isect_pos));
            float S2 = 0.5f * glm::length(glm::cross(vertices[0] - isect_pos, vertices[1] - isect_pos));
            glm::vec3 newNormal = glm::normalize(n0 * S0 / S + n1 * S1 / S + n2 * S2 / S);

            if ((glm::length(uvs[0]) != 0) && (glm::length(uvs[1]) != 0) && (glm::length(uvs[2]) != 0))
            {
                uv = uvs[0] * S0 / S + uvs[1] * S1 / S + uvs[2] * S2 / S;
            }

            normal = glm::normalize(multiplyMV(custom_obj.invTranspose, glm::vec4(newNormal, 0.f)));
            outside = glm::dot(normal, ray_inversed.direction) < 0;
            isect_pos = multiplyMV(custom_obj.transform, glm::vec4(isect_pos, 1.f));
        }
    }
    if (!outside)
    {
        normal = -normal;
    }
    return min_t;
}
