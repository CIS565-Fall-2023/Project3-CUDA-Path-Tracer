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
__host__ __device__ glm::vec3 getPointOnRay(const Ray& r, const float t) {
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(const glm::mat4& m, const glm::vec4& v) {
    return glm::vec3(m * v);
}

__host__ __device__ bool unitBoxIntersection(const Ray& q, float& tmin, float& tmax, glm::vec3& tmin_n, glm::vec3& tmax_n)
{
    tmin = -FLT_MAX;
    tmax = FLT_MAX;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = 1.f / q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) * qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) * qdxyz;
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
__host__ __device__ float boxIntersectionTest(const Geom& box, const Ray& r, glm::vec3& normal) {
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

        const glm::vec3 intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
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
__host__ __device__ float sphereIntersectionTest(const Geom& sphere, const Ray& r, glm::vec3& normal)
{
    const float radius = 0.5f;

    const glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    const glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float vDotDirection = glm::dot(ro, rd);
    float radicand = vDotDirection * vDotDirection - (glm::dot(ro, ro) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
    }
    else
    {
        t = max(t1, t2);
    }

    const glm::vec3 objspaceIntersection = ro + rd * t;

    const glm::vec3 intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool intersectAABB(const glm::vec3& ro, const glm::vec3& rdR, const glm::vec3& bboxMin, const glm::vec3& bboxMax, float& tMin)
{
    float tx1 = (bboxMin.x - ro.x) * rdR.x, tx2 = (bboxMax.x - ro.x) * rdR.x;
    tMin = min(tx1, tx2); float tMax = max(tx1, tx2);
    float ty1 = (bboxMin.y - ro.y) * rdR.y, ty2 = (bboxMax.y - ro.y) * rdR.y;
    tMin = max(tMin, min(ty1, ty2)), tMax = min(tMax, max(ty1, ty2));
    float tz1 = (bboxMin.z - ro.z) * rdR.z, tz2 = (bboxMax.z - ro.z) * rdR.z;
    tMin = max(tMin, min(tz1, tz2)), tMax = min(tMax, max(tz1, tz2));
    return tMax >= tMin && tMax > 0;
}

__host__ __device__ bool intersectBvh(const glm::vec3& ro, const glm::vec3& rd, const int nodeIdx, 
    const Triangle* const tris, const BvhNode* const bvhNodes, float& t, int& triIdx)
{
    int stack[32]; // assume BVH depth doesn't exceed 32
    int stackPtr = 0;
    stack[stackPtr++] = nodeIdx;
    t = FLT_MAX;
    triIdx = -1;

    const glm::vec3 rdR = 1.f / rd;

    while (stackPtr > 0)
    {
        const int currNodeIdx = stack[--stackPtr];
        const BvhNode& node = bvhNodes[currNodeIdx];

        float tmin;
        // if ray doesn't intersect or intersection point is past closest triangle so far, skip this subtree
        if (!intersectAABB(ro, rdR, node.aabb.bMin, node.aabb.bMax, tmin) || tmin > t)
        {
            continue;
        }

        if (node.isLeaf())
        {
            for (int i = 0; i < node.triCount; ++i)
            {
                const int potentialTriIdx = node.leftFirst + i;
                const Triangle& tri = tris[potentialTriIdx];
                const glm::vec3& v0 = tri.v0.pos;
                const glm::vec3& v1 = tri.v1.pos;
                const glm::vec3& v2 = tri.v2.pos;

                // I know barycentricPos.z == t but x and y are constantly giving me trouble, so I recalculate this at the end of meshIntersectionTest()
                glm::vec3 barycentricPos;
                if (!glm::intersectRayTriangle(ro, rd, v0, v1, v2, barycentricPos))
                {
                    continue;
                }

                if (barycentricPos.z < t)
                {
                    t = barycentricPos.z;
                    triIdx = potentialTriIdx;
                }
            }
        }
        else
        {
            // don't check intersections here unnecessarily, instead add to stack for now
            stack[stackPtr++] = node.leftFirst;
            stack[stackPtr++] = node.leftFirst + 1;
        }
    }

    return triIdx != -1;
}

__host__ __device__ glm::vec3 barycentric(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 p)
{
    const glm::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    const float d00 = glm::dot(v0, v0);
    const float d01 = glm::dot(v0, v1);
    const float d11 = glm::dot(v1, v1);
    const float d20 = glm::dot(v2, v0);
    const float d21 = glm::dot(v2, v1);
    const float invDenom = 1.f / (d00 * d11 - d01 * d01);
    const float v = (d11 * d20 - d01 * d21) * invDenom;
    const float w = (d00 * d21 - d01 * d20) * invDenom;
    const float u = 1.0f - v - w;
    return glm::vec3(u, v, w);
}

__host__ __device__ float meshIntersectionTest(const Geom& geom, const Triangle* const tris, const BvhNode* const bvhNodes,
    const Ray& r, glm::vec3& normal, glm::vec2& uv, int& triIdx, const bool useBvh)
{
    const glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    const glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t;
#if BVH_TOGGLE
    if (useBvh)
    {
        if (!intersectBvh(ro, rd, geom.bvhRootNodeIdx, tris, bvhNodes, t, triIdx))
        {
            return -1;
        }
    }
    else
    {
        t = FLT_MAX;
        triIdx = -1;

        for (int i = geom.startTriIdx; i < geom.startTriIdx + geom.numTris; ++i)
        {
            const Triangle& tri = tris[i];

            glm::vec3 barycentricPos;
            if (!glm::intersectRayTriangle(ro, rd, tri.v0.pos, tri.v1.pos, tri.v2.pos, barycentricPos))
            {
                continue;
            }

            if (barycentricPos.z < t)
            {
                t = barycentricPos.z;
                triIdx = i;
            }
        }

        if (triIdx == -1)
        {
            return -1;
        }
    }
#else
    if (!intersectBvh(ro, rd, geom.bvhRootNodeIdx, tris, bvhNodes, t, triIdx))
    {
        return -1;
    }
#endif

    const Triangle& tri = tris[triIdx];
    const Vertex& v0 = tri.v0;
    const Vertex& v1 = tri.v1;
    const Vertex& v2 = tri.v2;

    const glm::vec3 objSpaceIntersection = ro + rd * t;
    const glm::vec3 barycentricPos = barycentric(v0.pos, v1.pos, v2.pos, objSpaceIntersection);
    const glm::vec3 objSpaceNormal = glm::normalize(barycentricPos.x * v0.nor + barycentricPos.y * v1.nor + barycentricPos.z * v2.nor);

    const glm::vec3 intersectionPoint = multiplyMV(geom.transform, glm::vec4(objSpaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(objSpaceNormal, 0.f)));
    uv = barycentricPos.x * v0.uv + barycentricPos.y * v1.uv + barycentricPos.z * v2.uv;

    return glm::length(r.origin - intersectionPoint);
}
