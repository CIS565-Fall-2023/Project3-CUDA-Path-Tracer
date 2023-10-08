#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define RAY_MESH_AABB_INTERSECT_OPTIMISATION 1

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
        //normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/// <summary>
/// Returns true if the ray intersects with AABB. Both ray and AABB are expected to be in local-space.
/// </summary>
/// <param name="r"></param>
/// <param name="aabbMin"></param>
/// <param name="aabbMax"></param>
/// <returns></returns>
__host__ __device__ bool raycastAABB(const Ray& r, const glm::vec3& aabbMin, const glm::vec3& aabbMax)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float dirComponent = r.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (aabbMin[xyz] - r.origin[xyz]) / dirComponent;
            float t2 = (aabbMax[xyz] - r.origin[xyz]) / dirComponent;
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
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
        }
        return true;
    }
    return false;
}

__host__ __device__ float getTriArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
{
    //p1.z = 0;
    //p2.z = 0;
    //p3.z = 0;

    glm::vec3 v1 = p1 - p3;
    glm::vec3 v2 = p2 - p3;

    glm::vec3 crossedV = glm::cross(v1, v2); // cross product
    float area = glm::length(crossedV); // *0.5f; triangle area IS 1/2 of cross product but in our only use case this is cancelled out
    return area;
}

__host__ __device__ glm::vec3 getBarycentricInfluence(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3)
{
    glm::vec3 influence;
    float S = getTriArea(t1, t2, t3);
    float S1 = getTriArea(p, t1, t2);       // (0,1)
    float S2 = getTriArea(p, t1, t3);       // (0,2)
    float S3 = getTriArea(p, t2, t3);       // (1,2)

    influence.x = S3 / S;
    influence.y = S2 / S;
    influence.z = S1 / S;

    return influence;
}

__host__ __device__ float geomIntersectionTest(Geom mesh, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, Triangle* tris)
{
    glm::vec3 minBary = glm::vec3(FLT_MAX);
    glm::vec3 bary;
    int triIdx = -1;

    const glm::vec3 rayOriginLocal = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    const glm::vec3 rayDirLocal = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

#if RAY_MESH_AABB_INTERSECT_OPTIMISATION
    Ray localSpaceRay;
    localSpaceRay.origin = rayOriginLocal;
    localSpaceRay.direction = rayDirLocal;

    // First, check if the ray intersects with the AABB of the mesh.
    // If not, there is no need to further check through each ray
    if (!raycastAABB(localSpaceRay, mesh.aabbMin, mesh.aabbMax))
    {
        // No intersection with AABB
        // Won't intersect with mesh
        return -1;
    }
#endif

    // Go through every single triangle
    for (int i = mesh.startTriIdx; i <= mesh.endTriIdx; i++)
    {
        if (glm::intersectRayTriangle(rayOriginLocal, rayDirLocal, tris[i].v0.pos, tris[i].v1.pos, tris[i].v2.pos, bary))
        {
            if (bary.z < minBary.z)
            {
                minBary = bary;
                triIdx = i;
            }
        }
    }

    if (triIdx >= 0 && minBary.z < FLT_MAX)
    {
        glm::vec3 localIntersectionPt = rayOriginLocal + rayDirLocal * minBary.z;

        Triangle& tri = tris[triIdx];

        glm::vec3 localNor(0.0f);
        // Are vertex normals present?
        if (tri.hasNormals)
        {
            // Use barycentric coordinates to interpolate vertex normals at the intersection point
            //float u = minBary.x;
            //float v = minBary.y;
            //float w = 1.0f - u - v;

            //localNor = glm::normalize(u * tri.v0.nor + v * tri.v1.nor + w * tri.v2.nor);
            // whatever barycentric influence is returned by glm::intersectRayTriangle() is either wrong
            // or uvw are not what I think they are
            // normal calculation with those values is broken
            // so I'll recalculate barycentric influence here.
            const glm::vec3 barycentricWeights = getBarycentricInfluence(localIntersectionPt, tri.v0.pos, tri.v1.pos, tri.v2.pos);
            localNor = glm::normalize(barycentricWeights.x * tri.v0.nor + barycentricWeights.y * tri.v1.nor + barycentricWeights.z * tri.v2.nor);
        }
        else
        {
            // Cross product of triangle edges
            // This will result in flat shading
            // This is just a fallback for when the GLTF file doesn't have normal data
            glm::vec3 d1 = glm::normalize(tri.v1.pos - tri.v0.pos);
            glm::vec3 d2 = glm::normalize(tri.v2.pos - tri.v0.pos);
            localNor = glm::normalize(glm::cross(d1, d2));
        }

        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(localIntersectionPt, 1.0f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(localNor, 0.0f)));
    }

    return triIdx == -1 ? -1 : glm::length(intersectionPoint - r.origin);
}
