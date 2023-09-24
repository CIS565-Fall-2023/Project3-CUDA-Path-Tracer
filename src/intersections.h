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
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ bool unitBoxIntersection(const Ray& q, float& tmin, float& tmax, glm::vec3& tmin_n, glm::vec3& tmax_n)
{
    tmin = -1e38f;
    tmax = 1e38f;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    return tmax >= tmin && tmax > 0;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal) {
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin;
    float tmax;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    if (unitBoxIntersection(q, tmin, tmax, tmin_n, tmax_n)) 
    {
        if (tmin <= 0) 
        {
            tmin = tmax;
            tmin_n = tmax_n;
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
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal) {
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
    } else {
        t = max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool intersectAABB(const Ray& ray, const glm::vec3 bboxMin, const glm::vec3 bboxMax, float& tmin)
{
    float tx1 = (bboxMin.x - ray.origin.x) / ray.direction.x, tx2 = (bboxMax.x - ray.origin.x) / ray.direction.x;
    tmin = min(tx1, tx2); float tmax = max(tx1, tx2);
    float ty1 = (bboxMin.y - ray.origin.y) / ray.direction.y, ty2 = (bboxMax.y - ray.origin.y) / ray.direction.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bboxMin.z - ray.origin.z) / ray.direction.z, tz2 = (bboxMax.z - ray.origin.z) / ray.direction.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    return tmax >= tmin && tmax > 0;
}

__host__ __device__ bool intersectBvh(const Ray& ray, const int nodeIdx, const Triangle* tris, const BvhNode* bvhNodes, const int* bvhTriIdx, float& t, int& triIdx)
{
    int stack[32]; // assume BVH depth doesn't exceed 32
    int stackPtr = 0;
    stack[stackPtr++] = nodeIdx;
    bool intersects = false;
    t = FLT_MAX;

    while (stackPtr > 0)
    {
        const int currNodeIdx = stack[--stackPtr];
        const BvhNode& node = bvhNodes[currNodeIdx];

        float tmin;
        if (!intersectAABB(ray, node.aabbMin, node.aabbMax, tmin) || tmin > t)
        {
            continue;
        }

        if (node.isLeaf())
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                const int potentialTriIdx = bvhTriIdx[node.leftFirst + i];
                const Triangle& tri = tris[potentialTriIdx];
                const glm::vec3& v0 = tri.v0.pos;
                const glm::vec3& v1 = tri.v1.pos;
                const glm::vec3& v2 = tri.v2.pos;

                glm::vec3 barycentricPos;
                if (!glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, barycentricPos))
                {
                    continue;
                }

                if (barycentricPos.z < t)
                {
                    t = barycentricPos.z;
                    triIdx = potentialTriIdx;
                    intersects = true;
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.leftFirst;
            stack[stackPtr++] = node.leftFirst + 1;
        }
    }

    return intersects;
}

__host__ __device__ float meshIntersectionTest(Geom geom, Mesh mesh, Triangle* tris, BvhNode* bvhNodes, int* bvhTriIdx,
    Ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray transformedRay = { ro, rd };

    float t;
    int triIdx;
    if (!intersectBvh(transformedRay, mesh.bvhRootNode, tris, bvhNodes, bvhTriIdx, t, triIdx))
    {
        return -1;
    }

    const Triangle& tri = tris[triIdx];
    const glm::vec3& v0 = tri.v0.pos;
    const glm::vec3& v1 = tri.v1.pos;
    const glm::vec3& v2 = tri.v2.pos;

    glm::vec3 objSpaceIntersection = ro + rd * t;
    glm::vec3 objSpaceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(objSpaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(objSpaceNormal, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}
