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

__host__ __device__ bool bboxIntersectionTest(Ray r, AABB bbox)
{   
    glm::vec3 bmin = bbox.min;
    glm::vec3 bmax = bbox.max;
    float tmin = -FLT_MAX;
    float tmax = FLT_MAX;
    for (int i = 0; i < 3; i++)
    {
        float t1 = (bmin[i] - r.origin[i]) / r.direction[i];
        float t2 = (bmax[i] - r.origin[i]) / r.direction[i];
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }
    return tmax > max(tmin, 0.0f);;
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
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
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

/**
 * Check mesh intersection, need to loop through every triangle in the mesh
 */

__host__ __device__ float meshIntersectionTest(Geom geo, Triangle* triangles, int start, int end,
    AABB bbox, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal,
    glm::vec2& uv, bool& outside)
{
    float closest = FLT_MAX;
    Ray q;
    q.origin = multiplyMV(geo.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geo.inverseTransform, glm::vec4(r.direction, 0.0f)));
    
    if (!bboxIntersectionTest(q, bbox))
    {
        return -1;
    }
    for (int i = start; i < end; i = i + 1)
    {
        glm::vec3 v1 = triangles[i].v1;
        glm::vec3 v2 = triangles[i].v2;
        glm::vec3 v3 = triangles[i].v3;

        glm::vec3 n1 = triangles[i].n1;
        glm::vec3 n2 = triangles[i].n2;
        glm::vec3 n3 = triangles[i].n3;

        glm::vec3 ratioPoint;
        bool intersect = glm::intersectRayTriangle(q.origin, q.direction, v1, v2, v3, ratioPoint);
        //compute the barycentric location that fit normal
        glm::vec3 objPos = (1.0f - ratioPoint.x - ratioPoint.y) * v1 + ratioPoint.x * v2 + ratioPoint.y * v3;
        //get the barycentric intersection location in the world space
        glm::vec3 worldPos = multiplyMV(geo.transform, glm::vec4(objPos, 1.0f));
        if (!intersect)
        {
            continue;
        }
        else
        {
            float distance = glm::length(r.origin - worldPos);
            // if intersect with further point, don't check
            if (distance >= closest)
            {
                continue;
            }
            else
            {
                closest = distance;

                glm::vec2 uv1 = triangles[i].uv1;
                glm::vec2 uv2 = triangles[i].uv2;
                glm::vec2 uv3 = triangles[i].uv3;

                // compute barycentric coordinates to interpolate normal
                glm::vec3 edge1 = v1 - v2;
                glm::vec3 edge2 = v3 - v2;
                glm::vec3 nor = glm::cross(edge1, edge2);
                float dim = glm::length(nor);

                glm::vec3 e11 = v2 - objPos;
                glm::vec3 e12 = v3 - objPos;
                float u = glm::length(glm::cross(e11, e12));

                glm::vec3 e21 = v1 - objPos;
                glm::vec3 e22 = v3 - objPos;
                float v = glm::length(glm::cross(e21, e22));

                glm::vec3 e31 = v1 - objPos;
                glm::vec3 e32 = v2 - objPos;
                float w = glm::length(glm::cross(e31, e32));

                u /= dim;
                v /= dim;
                w /= dim;

                //compute whether it is outside or inside
                glm::vec3 invRay = -q.direction;
                float fac = glm::dot(invRay, normal);

                fac > 0 ? outside = true : outside = false;  
                normal = glm::normalize(u * n1 + v * n2 + w * n3);
                normal = glm::normalize(multiplyMV(geo.invTranspose, glm::vec4(normal, 0.0f)));
                uv = u * uv1 + v * uv2 + w * uv3;
            }
        }
    }
    intersectionPoint = multiplyMV(geo.transform, glm::vec4(getPointOnRay(q, closest), 1.0f));
    if (!outside)
    {
        normal = -normal;
    }
    return closest;
}

// check the distance between the ray origin (in object space) and the triangle
__device__ float triangleIntersectionTest(Geom geo, Triangle tri, Ray worldRay, Ray objectRay, glm::vec3& worldIntersect, glm::vec3& objectIntersect)
{
    glm::vec3 ratioPoint;
    bool intersect = glm::intersectRayTriangle(objectRay.origin, objectRay.direction, tri.v1, tri.v2, tri.v3, ratioPoint);
    if (!intersect)
        return FLT_MAX;
    //compute the barycentric location that fit normal
    objectIntersect = (1.0f - ratioPoint.x - ratioPoint.y) * tri.v1 + ratioPoint.x * tri.v2 + ratioPoint.y * tri.v3;
    //get the barycentric intersection location in the world space
    worldIntersect = multiplyMV(geo.transform, glm::vec4(objectIntersect, 1.0f));
    float distance = glm::length(worldRay.origin - worldIntersect);
    return distance;
}

__device__ void computeTriangleInfo(Geom geo, Triangle tri, glm::vec3& normal, glm::vec3 worldIntersect, glm::vec3 objectIntersect,
   Ray worldRay, Ray objectRay, glm::vec2& uv, bool& outside)
{
    // compute barycentric coordinates to interpolate normal
    glm::vec3 edge1 = tri.v1 - tri.v2;
    glm::vec3 edge2 = tri.v3 - tri.v2;
    glm::vec3 nor = glm::cross(edge1, edge2);
    float dim = glm::length(nor);

    glm::vec3 e11 = tri.v2 - objectIntersect;
    glm::vec3 e12 = tri.v3 - objectIntersect;
    float u = glm::length(glm::cross(e11, e12));

    glm::vec3 e21 = tri.v1 - objectIntersect;
    glm::vec3 e22 = tri.v3 - objectIntersect;
    float v = glm::length(glm::cross(e21, e22));

    glm::vec3 e31 = tri.v1 - objectIntersect;
    glm::vec3 e32 = tri.v2 - objectIntersect;
    float w = glm::length(glm::cross(e31, e32));

    u /= dim;
    v /= dim;
    w /= dim;

    //compute whether it is outside or inside
    glm::vec3 invRay = -objectRay.direction;
    float fac = glm::dot(invRay, normal);

    fac > 0 ? outside = true : outside = false;
    normal = glm::normalize(u * tri.n1 + v * tri.n2 + w * tri.n3);
    normal = glm::normalize(multiplyMV(geo.invTranspose, glm::vec4(normal, 0.0f)));
    uv = u * tri.uv1 + v * tri.uv2 + w * tri.uv3;
}

__device__ float traverseTree(BVHNode* nodes, Geom geo, Triangle* triangles,
    int start, int end, AABB bbox, Ray worldRay, glm::vec3& intersectionPoint,
    glm::vec3& normal, glm::vec2& uv, bool& outside, int meshInd)
{
    // record the closest intersection
    float closest = FLT_MAX;
    Ray objectRay;
    objectRay.origin = multiplyMV(geo.inverseTransform, glm::vec4(worldRay.origin, 1.0f));
    objectRay.direction = glm::normalize(multiplyMV(geo.inverseTransform, glm::vec4(worldRay.direction, 0.0f)));

    glm::vec3 worldIntersect = glm::vec3(0.f);
    glm::vec3 objectIntersect = glm::vec3(0.f);

    if (!bboxIntersectionTest(objectRay, bbox))
    {
        return -1;
    }
    int bvhStart = 2 * start - geo.meshid;
    int stack[64];
    int stackPtr = 0;
    int bvhPtr = bvhStart;
    stack[stackPtr++] = bvhStart;
    while(stackPtr)
    {
        bvhPtr = stack[--stackPtr];
        BVHNode currentNode = nodes[bvhPtr];
        // all the left and right indexes are 0
        BVHNode leftChild = nodes[currentNode.leftIndex + bvhStart];
        BVHNode rightChild = nodes[currentNode.rightIndex + bvhStart];

        bool hitLeft = bboxIntersectionTest(objectRay, leftChild.bbox);
        bool hitRight = bboxIntersectionTest(objectRay, rightChild.bbox);
        if (hitLeft)
        {
            // check triangle intersection
            if (leftChild.isLeaf == 1)
            {
                glm::vec3 tmpWorldIntersect = glm::vec3(0.f);
                glm::vec3 tmpObjectIntersect = glm::vec3(0.f);
                float distance = triangleIntersectionTest(geo, triangles[leftChild.triangleIndex + start], worldRay, objectRay, tmpWorldIntersect, tmpObjectIntersect);
                // if is closer, then calculate normal and uv
                if (distance < closest)
                {
                    computeTriangleInfo(geo, triangles[leftChild.triangleIndex + start], normal, tmpWorldIntersect, tmpObjectIntersect,
                        worldRay, objectRay, uv, outside);
                    closest = distance;
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.leftIndex + bvhStart;
            }
            
        }
        if (hitRight)
        {
            // check triangle intersection
            if (rightChild.isLeaf == 1)
            {
                
                glm::vec3 tmpWorldIntersect = glm::vec3(0.f);
                glm::vec3 tmpObjectIntersect = glm::vec3(0.f);
                float distance = triangleIntersectionTest(geo, triangles[rightChild.triangleIndex + start], worldRay, objectRay, tmpWorldIntersect, tmpObjectIntersect);
                // if is closer, then calculate normal and uv
                if (distance < closest)
                {
                    computeTriangleInfo(geo, triangles[rightChild.triangleIndex + start], normal, tmpWorldIntersect, tmpObjectIntersect,
                        worldRay, objectRay, uv, outside);
                    closest = distance;
                }
            }
            else
            {
                stack[stackPtr++] = currentNode.rightIndex + bvhStart;
            }
            
        }
    }
    intersectionPoint = multiplyMV(geo.transform, glm::vec4(getPointOnRay(objectRay, closest), 1.0f));
    if (!outside)
    {
        normal = -normal;
    }
    return closest;
}

