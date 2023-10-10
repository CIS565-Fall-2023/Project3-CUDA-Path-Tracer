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
            if (ta > -0.0001f && ta > tmin) {
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
        if (tmin <= -0.0001f) {
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
    } else if (t1 > 0 && t2 > -0.0001f) {
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


__host__ __device__ float triIntersectionTest(Geom geom, Ray r,
        glm::vec3 &intersectionPoint,glm::vec2 &uv,glm::vec3 &dpdu,glm::vec3 &dpdv,  glm::vec3 &normal, bool &outside) {

    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    Triangle &triangle=geom.tri;
    //refenrenced from wikipedia: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    outside=true;
    float a, f, u, v;
    glm::vec3 edge1=triangle.vertices[1]- triangle.vertices[0];
    glm::vec3 edge2=triangle.vertices[2]- triangle.vertices[0];
    float len1 = glm::length(triangle.vertices[1]);
    float len2 = glm::length(edge2);
    glm::vec3 h=glm::cross(rt.direction, edge2);
    float lenh = glm::length(h);
    a = glm::dot(edge1,h);

    if (a > -0.0000001f && a < 0.0000001f) {
        return -1;    // This ray is parallel to this triangle.
    }

    f = 1.0f / a;
    glm::vec3 s=rt.origin-triangle.vertices[0];
    u = f * (glm::dot(s,h));

    if (u < 0.0 || u > 1.0) {
        return -1;
    }

    glm::vec3 q=glm::cross(s, edge1);
    v = f *glm::dot(rt.direction,q);

    if (v < 0.0 || u + v > 1.0) {
        return -1;
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f *glm::dot(edge2,q);
    if (t > 0.0000001f) // ray intersection
    {
        intersectionPoint = getPointOnRay(rt, t);
        normal=(1.0f-u-v)*triangle.normals[0]+u*triangle.normals[1]+v*triangle.normals[2];
        uv=(1.0f-u-v)*triangle.uvs[0]+u*triangle.uvs[1]+v*triangle.uvs[2];
        //normal=triangle.g_norm;
        if(glm::dot(triangle.g_norm,rt.direction)>0.0f){
            outside=false;
        }
    } else // This means that there is a line intersection but not a ray intersection.
    {
        return -1;
    }
    dpdu=glm::normalize(multiplyMV(geom.transform, glm::vec4(triangle.dpdu, 0.f)));
    dpdv=glm::normalize(multiplyMV(geom.transform, glm::vec4(triangle.dpdv, 0.f)));
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(intersectionPoint, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));
    if(outside==false){
        normal=-normal;
    }
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float aabbIntersectionTest(BVHnode box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

    float tmin = -1e38f;
    float tmax = 1e38f;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = r.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ //{
            float t1 = (box.min[xyz] - r.origin[xyz]) / qdxyz;
            float t2 = (box.max[xyz] - r.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > -0.0001f && ta > tmin) {
                tmin = ta;
            }
            if (tb < tmax) {
                tmax = tb;
            }
        //}
    }

    if (tmax >= tmin && tmax > 0) {
        if (tmin <= -0.0001f) 
            tmin = tmax;
        return tmin;
    }
    return -1;
}
