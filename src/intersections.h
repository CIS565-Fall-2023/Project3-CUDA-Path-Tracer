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

__device__ bool intersectTriangle(const Triangle & tri, Ray& r, ShadeableIntersection* isect) {
    glm::vec3 bary;
    bool hasHit = glm::intersectRayTriangle(r.origin, r.direction, tri.p1, tri.p2, tri.p3, bary);
    if (hasHit) {
        glm::vec3 intersection = (1.0f - bary.x - bary.y) * tri.p1 + bary.x * tri.p2 + bary.y * tri.p3;
        float t = bary.z;
        if (r.min_t < t && t < r.max_t) {
            r.max_t = t;
            isect->t = t;
            isect->materialId = tri.materialID;
            isect->intersectionPoint = r.at(t);
            isect->uv = (1.0f - bary.x - bary.y) * tri.uv1 + bary.x * tri.uv2 + bary.y * tri.uv3;
            isect->surfaceNormal = glm::normalize((1.0f - bary.x - bary.y) * tri.n1 + bary.x * tri.n2 + bary.y * tri.n3);
            isect->primitive = &tri;
        }
    }
    return hasHit;
}


__device__ bool intersectAABB(const Ray& ray, const AABB& aabb)
{
    glm::vec3 invDirection = 1.0f / ray.direction;
    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;
    /* Unrolling the loop seems not affecting the FPS a lot? */
#pragma unroll
    for (int axis = 0; axis < 3; axis++)
    {
        float t1 = (aabb.pmin[axis] - ray.origin[axis]) * invDirection[axis];
        float t2 = (aabb.pmax[axis] - ray.origin[axis]) * invDirection[axis];
        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }
        tmin = glm::max(tmin, t1);
        tmax = glm::min(tmax, t2);
        if (tmin > tmax) return false;
    }
    return !(tmin > ray.max_t || tmax < ray.min_t);
}


__device__ bool intersectBVH(const BVHNode* nodes, int nodeCount, const Triangle* tris, Ray& ray, ShadeableIntersection& intersection)
{
    int nodeIndex = 0;
    while (nodeIndex < nodeCount) {
        const BVHNode& node = nodes[nodeIndex];
        if (intersectAABB(ray, node.aabb)) {
            if (node.isLeaf) {
                for (size_t i = node.startPrim; i < node.endPrim; i++)
                {
                    intersectTriangle(tris[i], ray, &intersection);
                }
            }
            nodeIndex = node.hit;
        }
        else {
            nodeIndex = node.miss;
        }
    }
    return intersection.t > EPSILON;
}