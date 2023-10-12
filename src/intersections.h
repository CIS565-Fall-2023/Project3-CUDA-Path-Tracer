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

__host__ __device__ float meshIntersectionTest(
    const Geom &geom,
    const Mesh *meshes, const glm::vec3 *vertices, const glm::vec3 *normals, const glm::vec2 *texcoords,
    const Ray r,
    glm::vec3 &intersectionPoint, int &materialid, glm::vec3 &normal, glm::vec2 &texcoord)
{
    float t_min = FLT_MAX;
    for (int i = geom.meshidx; i < geom.meshidx + geom.meshcnt; i++)
    {
        Mesh m = meshes[i];
        glm::vec3 v0 = vertices[m.v[0]];
        glm::vec3 v1 = vertices[m.v[1]];
        glm::vec3 v2 = vertices[m.v[2]];

        glm::vec3 baryPosition;

        if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, baryPosition))
        {
            glm::vec3 point = (1 - baryPosition.x - baryPosition.y) * v0 +
                              baryPosition.x * v1 +
                              baryPosition.y * v2;
            float t = glm::length(r.origin - point);
            if (t > 0.0f && t < t_min)
            {
                t_min = t;
                intersectionPoint = point;
                materialid = m.materialid;
                if (m.vn[0] == -1 || m.vn[1] == -1 || m.vn[2] == -1)
                    normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                else
                    normal = (1 - baryPosition.x - baryPosition.y) * normals[m.vn[0]] +
                             baryPosition.x * normals[m.vn[1]] +
                             baryPosition.y * normals[m.vn[2]];
                if (m.vt[0] == -1 || m.vt[1] == -1 || m.vt[2] == -1)
                    texcoord = glm::vec2(-1.f); // no texture
                else
                    texcoord = (1 - baryPosition.x - baryPosition.y) * texcoords[m.vt[0]] +
                               baryPosition.x * texcoords[m.vt[1]] +
                               baryPosition.y * texcoords[m.vt[2]];
            }
        }
    }
    return t_min;
}

// reference https://tavianator.com/2011/ray_box.html
__host__ __device__ bool aabbIntersectionTest(const AABB &aabb, const Ray &ray)
{
    float invDirX = 1.f / ray.direction.x;
    float invDirY = 1.f / ray.direction.y;
    float invDirZ = 1.f / ray.direction.z;

    float tx1 = (aabb.min.x - ray.origin.x) * invDirX;
    float tx2 = (aabb.max.x - ray.origin.x) * invDirX;

    float tmin = glm::min(tx1, tx2);
    float tmax = glm::max(tx1, tx2);

    float ty1 = (aabb.min.y - ray.origin.y) * invDirY;
    float ty2 = (aabb.max.y - ray.origin.y) * invDirY;

    tmin = glm::max(tmin, glm::min(ty1, ty2));
    tmax = glm::min(tmax, glm::max(ty1, ty2));

    float tz1 = (aabb.min.z - ray.origin.z) * invDirZ;
    float tz2 = (aabb.max.z - ray.origin.z) * invDirZ;

    tmin = glm::max(tmin, glm::min(tz1, tz2));
    tmax = glm::min(tmax, glm::max(tz1, tz2));

    return tmax >= glm::max(0.f, tmin);
}

__host__ __device__ void finalIntersectionTest(
    const Mesh &m, const glm::vec3 *vertices, const glm::vec3 *normals, const glm::vec2 *texcoords,
    const Ray &r,
    float &t_min, glm::vec3 &intersectionPoint, int &materialid, glm::vec3 &normal, glm::vec2 &texcoord)
{
    glm::vec3 v0 = vertices[m.v[0]];
    glm::vec3 v1 = vertices[m.v[1]];
    glm::vec3 v2 = vertices[m.v[2]];

    glm::vec3 baryPosition;

    if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, baryPosition))
    {
        glm::vec3 point = (1 - baryPosition.x - baryPosition.y) * v0 +
                          baryPosition.x * v1 +
                          baryPosition.y * v2;
        float t = glm::length(r.origin - point);
        if (t > 0.0f && t < t_min)
        {
            t_min = t;
            intersectionPoint = point;
            materialid = m.materialid;
            if (m.vn[0] == -1 || m.vn[1] == -1 || m.vn[2] == -1)
                normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
            else
                normal = (1 - baryPosition.x - baryPosition.y) * normals[m.vn[0]] +
                         baryPosition.x * normals[m.vn[1]] +
                         baryPosition.y * normals[m.vn[2]];
            if (m.vt[0] == -1 || m.vt[1] == -1 || m.vt[2] == -1)
                texcoord = glm::vec2(-1.f); // no texture
            else
                texcoord = (1 - baryPosition.x - baryPosition.y) * texcoords[m.vt[0]] +
                           baryPosition.x * texcoords[m.vt[1]] +
                           baryPosition.y * texcoords[m.vt[2]];
        }
    }
}

// reference https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
__host__ __device__ float meshIntersectionTestBVH(
    Geom &geom,
    BVHNode *bvh,
    Mesh *meshes, glm::vec3 *vertices, glm::vec3 *normals, glm::vec2 *texcoords,
    Ray r,
    glm::vec3 &intersectionPoint, int &materialid, glm::vec3 &normal, glm::vec2 &texcoord)
{
    float t_min = FLT_MAX;

    int stack[64];
    int *stackPtr = stack;
    *stackPtr++ = -1;

    int nodeIdx = geom.bvhrootidx;

    do
    {
        BVHNode node = bvh[nodeIdx];

        int idxL = node.left;
        BVHNode childL = bvh[idxL];
        int idxR = node.right;
        BVHNode childR = bvh[idxR];

        bool overlapL = aabbIntersectionTest(childL.aabb, r);
        bool overlapR = aabbIntersectionTest(childR.aabb, r);

        if (overlapL && (childL.left == -1 && childL.right == -1))
        {
            finalIntersectionTest(
                meshes[childL.meshidx], vertices, normals, texcoords,
                r,
                t_min, intersectionPoint, materialid, normal, texcoord);
        }

        if (overlapR && (childR.left == -1 && childR.right == -1))
        {
            finalIntersectionTest(
                meshes[childR.meshidx], vertices, normals, texcoords,
                r,
                t_min, intersectionPoint, materialid, normal, texcoord);
        }

        bool traverseL = overlapL && !(childL.left == -1 && childL.right == -1);
        bool traverseR = overlapR && !(childR.left == -1 && childR.right == -1);

        if (!traverseL && !traverseR)
            nodeIdx = *--stackPtr; // pop
        else
        {
            nodeIdx = (traverseL) ? idxL : idxR;
            if (traverseL && traverseR)
                *stackPtr++ = idxR; // push
        }

    }
    while (nodeIdx != -1);

    return t_min;
}
