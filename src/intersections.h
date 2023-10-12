#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <iostream>
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
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
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

__host__ __device__
bool boundingBoxIntersectionTest(AABB aabb, Ray q)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    for (int i = 0; i < 3; i++) {
        float d = q.direction[i];
        float t1 = (aabb.min[i] - q.origin[i]) / d;
        float t2 = (aabb.max[i] - q.origin[i]) / d;
        float tempMin = glm::min(t1, t2);
        float tempMax = glm::max(t1, t2);
        if (tempMin > 0 && tempMin > tmin) {
            tmin = tempMin;
        }
        if (tempMax < tmax) {
            tmax = tempMax;
        }
    }

    if (tmax >= tmin && tmax > 0) {       
        return true;
    }

    return false;
}

__host__ __device__ float objIntersectionTest(Geom mesh, Ray r, Triangle* triangles,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    Ray q = r;
 
    if (!boundingBoxIntersectionTest(mesh.aabb, q)) {
        return -1;
    }

    float tmin = 1e38f;
    int endIdx = mesh.triIdx + mesh.triCnt;
    bool hasIntersection = false;
    for (int i = mesh.triIdx; i < endIdx ; i++)
    {
        Triangle triangle = triangles[i];
        glm::vec3 baryPos;
        glm::vec3 objspaceIntersection;
        if (glm::intersectRayTriangle(q.origin, q.direction,
            triangle.v[0], triangle.v[1], triangle.v[2], baryPos))
        {
            objspaceIntersection = (1 - baryPos[0] - baryPos[1]) * triangle.v[0] +
                baryPos[0] * triangle.v[1] + baryPos[1] * triangle.v[2];

            float t = glm::length(objspaceIntersection - q.origin);
            if (t < tmin)
            {
                tmin = t;
                glm::vec3 a = triangle.v[1] - triangle.v[0];
                glm::vec3 b = triangle.v[2] - triangle.v[0];
                glm::vec3 localNorm = glm::normalize(glm::cross(a, b));
                outside = glm::dot(localNorm, q.direction) < 0;
                intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
                normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(localNorm, 0.f)));
                hasIntersection = true;
            }          
        }
    }

    if (hasIntersection)
    {
        return tmin;
    }
    
    return -1;
}


__host__ __device__ float IntersectAABB(Ray r, BVHNode node)
{
    float tx1 = (node.aabbMin.x - r.origin.x) * r.direction.x, tx2 = (node.aabbMax.x - r.origin.x) * r.direction.x;
    float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
    float ty1 = (node.aabbMin.y - r.origin.y) * r.direction.y, ty2 = (node.aabbMax.y - r.origin.y) * r.direction.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
    float tz1 = (node.aabbMin.z - r.origin.z) * r.direction.z, tz2 = (node.aabbMax.z - r.origin.z) * r.direction.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));
    if (tmax >= tmin && tmax > 0) return tmin; else return 1e30f;
}

__host__ __device__ float BVHIntersect(Ray r,
    Triangle* tri, BVHNode* bvhNode, int* triIdx,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int& geomIdx)
{
    BVHNode node = bvhNode[0];
    BVHNode stack[32];
    int stackPtr = 0;
    float tmin = 1e38f;
    bool hasIntersection = false;

    while (1)
    {
        if (node.triCount > 0) // isLeaf()
        {

            for (int i = 0; i < node.triCount; i++)
            {
                Triangle triangle = tri[triIdx[node.firstTriIdx + i]];
                glm::vec3 baryPos;
                glm::vec3 intersectionPos;
                if (glm::intersectRayTriangle(r.origin, r.direction,
                    triangle.v[0], triangle.v[1], triangle.v[2], baryPos))
                {
                    intersectionPos = (1 - baryPos[0] - baryPos[1]) * triangle.v[0] +
                        baryPos[0] * triangle.v[1] + baryPos[1] * triangle.v[2];

                    float t = glm::length(intersectionPos - r.origin);
                    if (t < tmin)
                    {
                        tmin = t;
                        glm::vec3 a = triangle.v[1] - triangle.v[0];
                        glm::vec3 b = triangle.v[2] - triangle.v[0];
                        glm::vec3 localNorm = glm::normalize(glm::cross(a, b));
                        outside = glm::dot(localNorm, r.direction) < 0;
                        intersectionPoint = intersectionPos;
                        normal = localNorm;
                        geomIdx = triangle.geomIdx;
                        hasIntersection = true;
                    }
                }

            }
            if (stackPtr == 0) {
                if (hasIntersection) return tmin;
                else return -1;
            }
            else {
                node = stack[--stackPtr];
            }
            continue;
        }

        BVHNode child1 = bvhNode[node.leftNode];
        BVHNode child2 = bvhNode[node.leftNode + 1];
        float dist1 = IntersectAABB(r, child1);
        float dist2 = IntersectAABB(r, child2);
        if (dist1 > dist2)
        {
            float d = dist1; dist1 = dist2; dist2 = d;
            BVHNode c = child1; child1 = child2; child2 = c;
        }
        if (dist1 == 1e30f)
        {
            if (stackPtr == 0) {
                if (hasIntersection) return tmin;
                else return -1;
            }
            else {
                node = stack[--stackPtr];
            }
        }
        else
        {
            node = child1;
            if (dist2 != 1e30f) stack[stackPtr++] = child2;
        }
    }
}


