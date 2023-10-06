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

__host__ __device__ float triangleIntersectionTest(Geom custom_obj, Ray r,
    glm::vec3& intersectionPoint, Triangle* triangles, int triangleIdStart, int triangleIdEnd,
    glm::vec3& normal, bool& outside, glm::vec2& uv)
{
    // Convert ray to local space
    Ray ray_inversed;
    ray_inversed.origin = multiplyMV(custom_obj.inverseTransform, glm::vec4(r.origin, 1.0f));
    ray_inversed.direction = glm::normalize(multiplyMV(custom_obj.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float min_t = FLT_MAX;

    for (int i = triangleIdStart; i < triangleIdEnd; i++)
    {
        const Triangle& triangle = triangles[i];

        glm::vec3 baryPos;

        if (glm::intersectRayTriangle(ray_inversed.origin, ray_inversed.direction,
            triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], baryPos))
        {
            glm::vec3 isect_pos = (1.f - baryPos.x - baryPos.y) * triangle.vertices[0] +
                baryPos.x * triangle.vertices[1] + baryPos.y * triangle.vertices[2];
            intersectionPoint = multiplyMV(custom_obj.transform, glm::vec4(isect_pos, 1.f));
            float t = glm::length(r.origin - intersectionPoint);

            if (t >= min_t)
            {
                continue;
            }
            min_t = t;

            glm::vec3 n0, n1, n2;
            if (glm::length(triangle.normals[0]) != 0 &&
                glm::length(triangle.normals[1]) != 0 &&
                glm::length(triangle.normals[2]) != 0)
            {
                n0 = triangle.normals[0];
                n1 = triangle.normals[1];
                n2 = triangle.normals[2];
            }
            else
            {
                n0 = glm::normalize(glm::cross(triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]));
                n1 = glm::normalize(glm::cross(triangle.vertices[2] - triangle.vertices[1], triangle.vertices[0] - triangle.vertices[1]));
                n2 = glm::normalize(glm::cross(triangle.vertices[0] - triangle.vertices[2], triangle.vertices[1] - triangle.vertices[2]));
            }

            // Barycentric Interpolation
            const glm::vec3 cross_v1v2_v1v3 = glm::cross(triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0]);
            float S = 0.5f * glm::length(cross_v1v2_v1v3);
            float S0 = 0.5f * glm::length(glm::cross(triangle.vertices[1] - isect_pos, triangle.vertices[2] - isect_pos));
            float S1 = 0.5f * glm::length(glm::cross(triangle.vertices[0] - isect_pos, triangle.vertices[2] - isect_pos));
            float S2 = S - S0 - S1;

            glm::vec3 newNormal = glm::normalize((n0 * S0 + n1 * S1 + n2 * S2) / S);
            normal = glm::normalize(multiplyMV(custom_obj.invTranspose, glm::vec4(newNormal, 0.f)));
            outside = glm::dot(normal, ray_inversed.direction) < 0;

            if (glm::length(triangle.uvs[0]) != 0 &&
                glm::length(triangle.uvs[1]) != 0 &&
                glm::length(triangle.uvs[2]) != 0)
            {
                uv = (triangle.uvs[0] * S0 + triangle.uvs[1] * S1 + triangle.uvs[2] * S2) / S;
            }
        }
    }

    if (!outside)
    {
        normal = -normal;
    }
    return min_t;
}


class OctreeNode {
public:
    glm::vec3 minCorner, maxCorner;  // Bounding box
    std::vector<Geom> objects;  // Objects in this node
    OctreeNode* children[8] = { nullptr };  // Pointers to children nodes
};

__host__ __device__ OctreeNode* buildOctree(const std::vector<Geom>& objects, glm::vec3 minCorner, glm::vec3 maxCorner, int depth) {
    if (objects.size() == 0 || depth <= 0) {
        return nullptr;
    }

    OctreeNode* node = new OctreeNode();
    node->minCorner = minCorner;
    node->maxCorner = maxCorner;

    if (objects.size() == 1 || depth == 1) {
        node->objects = objects;
        return node;
    }

    glm::vec3 center = (minCorner + maxCorner) / 2.0f;

    // Partition objects into 8 octants
    std::vector<Geom> octantObjects[8];

    for (const Geom& obj : objects) {
        glm::vec3 objPosition = obj.translation;

        // Determine the octant for the object based on its position relative to center
        int octantIndex =
            (objPosition.x >= center.x) * 4 +
            (objPosition.y >= center.y) * 2 +
            (objPosition.z >= center.z);
        octantObjects[octantIndex].push_back(obj);
    }

    // Recursive build children
    for (int i = 0; i < 8; ++i) {
        glm::vec3 childMinCorner = minCorner + glm::vec3((i & 4) * center.x, (i & 2) * center.y, (i & 1) * center.z);
        glm::vec3 childMaxCorner = center + glm::vec3((i & 4) * maxCorner.x, (i & 2) * maxCorner.y, (i & 1) * maxCorner.z);

        node->children[i] = buildOctree(octantObjects[i], childMinCorner, childMaxCorner, depth - 1);
    }

    return node;
}

