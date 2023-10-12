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
        // Commented out b/c affects refraction
        // This makes nor point inward if the ray starts inside, while
        // convention assumes nor would always point away from the sphere.
        //normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ void swap(float& a, float& b) {
    float tmp = a;
    a = b;
    b = tmp;
}

// reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
__host__ __device__ bool aabbIntersectionTest(AABB box, Ray& r, float& t) {
    float tmin = (box.min.x - r.origin.x) / r.direction.x;
    float tmax = (box.max.x - r.origin.x) / r.direction.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (box.min.y - r.origin.y) / r.direction.y;
    float tymax = (box.max.y - r.origin.y) / r.direction.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (box.min.z - r.origin.z) / r.direction.z;
    float tzmax = (box.max.z - r.origin.z) / r.direction.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    bool intersect = tzmin <= tzmax && tzmax >= 0;
    t = intersect ? tzmin : -1.f;
    if (t < 0.f) t = tzmax;

    return true;
}

/**
 * Test intersection between a ray and a triangulated gltf/glb/obj mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray& r,
    const Triangle* tris, glm::vec3& intersectionPoint, glm::vec3& normal) {

    Triangle tri;
    glm::vec3 barycenter, objSpaceIsect;
    float tMin = FLT_MAX;
    bool hasIntersection = false;

    // do intersection test with every triangle
    for (int i = mesh.startTriIdx; i < mesh.startTriIdx + mesh.triangleCount; ++i) {
        
        tri = tris[i];
        bool intersect = glm::intersectRayTriangle(r.origin, r.direction,
            tri.v0.pos, tri.v1.pos, tri.v2.pos, barycenter);
        if (!intersect) continue;

        // find intersection point and normal
        float u = barycenter.x;
        float v = barycenter.y;
        float w = 1.f - u - v;
        objSpaceIsect = w * tri.v0.pos
            + u * tri.v1.pos
            + v * tri.v2.pos;

        float t = glm::length(r.origin - objSpaceIsect);

        glm::vec3 intersectionNormal = glm::normalize(
            glm::cross(tri.v1.pos - tri.v0.pos, 
                       tri.v2.pos - tri.v0.pos));

        if (t < tMin) {
            tMin = t;
            intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objSpaceIsect, 1.f));
            normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(intersectionNormal, 0.f)));;
            hasIntersection = true;
        }
    }

    if (hasIntersection) {
        return tMin;
    }
    return -1.f;
}


__host__ __device__ bool isNodeLeaf(const BVHNode* node) {
    return (node->triCount > 0);
}

__host__ __device__ void bvhIntersectTriangles(const Geom* geom, const Triangle* tris, Ray& r,
    int start, int trisCount,
    glm::vec3& intersectionPoint, glm::vec3& normal, float& tMin) {
    
    glm::vec3 barycenter, objSpaceIsect;
    for (int i = start; i < start + trisCount; ++i) {
    
        bool intersect = glm::intersectRayTriangle(r.origin, r.direction,
            tris[i].v0.pos, tris[i].v1.pos, tris[i].v2.pos, barycenter);
        if (!intersect) continue;
        r.intersectionCount++;

        // find intersection point and normal
        float u = barycenter.x;
        float v = barycenter.y;
        float w = 1.f - u - v;
        objSpaceIsect = w * tris[i].v0.pos
            + u * tris[i].v1.pos
            + v * tris[i].v2.pos;

        float t = glm::length(r.origin - objSpaceIsect);
        glm::vec3 intersectionNormal = glm::normalize(
            glm::cross(tris[i].v1.pos - tris[i].v0.pos,
                       tris[i].v2.pos - tris[i].v0.pos));

        if (t < tMin)
        {
            tMin = t;
            intersectionPoint = objSpaceIsect; multiplyMV(geom->transform, glm::vec4(objSpaceIsect, 1.f));
            normal = intersectionNormal; glm::normalize(multiplyMV(geom->invTranspose, glm::vec4(intersectionNormal, 0.f)));
        }
    }
}


/**
 * Test intersection between a ray and a BVH structure.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 * 
 * reference:
 * https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/
 */
__host__ __device__ float bvhIntersectionTest(Geom geom, const BVHNode* nodes,
    const Triangle* tris, Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal) {

    float tMin = FLT_MAX;
    int stack[32];
    int stackPtr = -1;

    // Push root node
    stack[++stackPtr] = 0;
    int currNodeIdx = stack[stackPtr];

    while (stackPtr >= 0)
    {
        // Check intersection with left and right children
        int leftChild = nodes[currNodeIdx].leftChild;
        int rightChild = nodes[currNodeIdx].rightChild;
        const BVHNode* left = &nodes[leftChild];
        const BVHNode* right = &nodes[rightChild];

        float t;
        bool intersectLeft = aabbIntersectionTest(left->aabb, r, t);
        bool intersectRight = aabbIntersectionTest(right->aabb, r, t);

        // If intersection found, and they are leaf nodes, check for triangle intersections
        if (intersectLeft && isNodeLeaf(left)) {
            bvhIntersectTriangles(&geom, tris, r, left->firstTriIdx, left->triCount, intersectionPoint, normal, tMin);
            //if (r.intersectionCount >= triangleCount) return -1.f;
        }
        if (intersectRight && isNodeLeaf(right)) {
            bvhIntersectTriangles(&geom, tris, r, right->firstTriIdx, right->triCount, intersectionPoint, normal, tMin);
            //if (r.intersectionCount >= triangleCount) return -1.f;
        }

        // If internal nodes, keep traversing
        bool traverseLeftSubtree = (intersectLeft && !isNodeLeaf(left));
        bool traverseRightSubtree = (intersectRight && !isNodeLeaf(right));

        if (!traverseLeftSubtree && !traverseRightSubtree) {
            // Pop node from stack
            currNodeIdx = stack[stackPtr--];
        }
        else {
            currNodeIdx = traverseLeftSubtree ? leftChild : rightChild;
            if (traverseLeftSubtree && traverseRightSubtree) {
                // Push right child onto stack
                stack[++stackPtr] = rightChild;
            }
        }
    }

    return tMin;
}