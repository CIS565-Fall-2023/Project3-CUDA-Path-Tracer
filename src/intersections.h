#pragma once

#include <glm/glm.hpp>

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
__host__ __device__ glm::vec3 multiplyMV(const glm::mat4& m, const glm::vec4& v) {
    return glm::vec3(m * v);
}


/*
* return val:
* x: t
* y: v0's weight
* z: v1's weight
*/
__host__ __device__ float3 triangleIntersectionTest(TriangleDetail triangle, Ray r)
{
    glm::vec3 v0 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v0, 1.f));
    glm::vec3 v1 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v1, 1.f));
    glm::vec3 v2 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v2, 1.f));
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 tvec = r.origin - v0;
    glm::vec3 pvec = glm::cross(r.direction, v0v2);
    float det = dot(v0v1, pvec);
    if (fabs(det) < 1e-5)
        return { -1,0,0 };
    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float invDet = 1.0 / det;
    float t = dot(qvec, v0v2) * invDet;
    float t_min = -1e38f;
    float t_max = 1e38f;
    if (t >= t_max || t <= t_min)
        return { -1,0,0 };
    float u = dot(tvec, pvec) * invDet;
    float v = dot(r.direction, qvec) * invDet;
    if (v < 0 || u + v > 1 || u < 0 || u > 1)
        return { -1,0,0 };
    return { t, 1 - u - v, u };
}

__device__ bool intersectTBB(const Ray& ray, const TBB& aabb, float& tmin) {
    glm::vec3 invDir = 1.0f / ray.direction;

    float t1 = (aabb.min.x - ray.origin.x) * invDir.x;
    float t2 = (aabb.max.x - ray.origin.x) * invDir.x;
    float t3 = (aabb.min.y - ray.origin.y) * invDir.y;
    float t4 = (aabb.max.y - ray.origin.y) * invDir.y;
    float t5 = (aabb.min.z - ray.origin.z) * invDir.z;
    float t6 = (aabb.max.z - ray.origin.z) * invDir.z;
    float tmin_temp = glm::max(glm::max(glm::min(t1, t2), glm::min(t3, t4)), glm::min(t5, t6));
    float tmax_temp = glm::min(glm::min(glm::max(t1, t2), glm::max(t3, t4)), glm::max(t5, t6));
    if (tmax_temp < 0 || tmin_temp > tmax_temp)
        return false;
    if (tmin_temp > 0)
        tmin = tmin_temp;
    else
        tmin = tmax_temp;

    return true;
}

__device__ __inline__ int mapDirToIdx(cudaTextureObject_t cubemap, const glm::vec3& direction) {
    return texCubemap<int>(cubemap, direction.x, direction.y, direction.z);
}

__device__ float3 sceneIntersectionTest(TriangleDetail* triangles, TBVHNode* nodes, int nodesNum, Ray r, int& hit_geom_index, cudaTextureObject_t cubemap) {
    int tbbOffset = mapDirToIdx(cubemap, r.direction);
    int currentNodeIdx = 0;
    float t_min = FLT_MAX;
    float3 tmp_t;
    float3 t;
    hit_geom_index = -1;

    while (currentNodeIdx != -1 && currentNodeIdx < nodesNum)
    {
        float tbbTmin = FLT_MAX;
        const TBVHNode& currNode = nodes[tbbOffset * nodesNum + currentNodeIdx];
        if (intersectTBB(r, currNode.tbb, tbbTmin) && tbbTmin <= t_min) {
            if (currNode.isLeaf)
            {
                t = triangleIntersectionTest(triangles[currNode.triId], r);
                if (t.x > 0.0f && t_min > t.x) {
                    tmp_t = t;
                    t_min = t.x;
                    hit_geom_index = currNode.triId;
                }
            }
            currentNodeIdx++;
        }
        else
        {
            currentNodeIdx = currNode.miss;
        }
    }
    return tmp_t;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}