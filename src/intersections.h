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

__host__ __device__ float closestPointOnCube(Geom c, glm::vec2 xi, glm::vec3 isect_point, glm::vec3& hit_point, Ray& light, float r) {
    glm::vec3 p_local = multiplyMV(c.inverseTransform, glm::vec4(isect_point, 1.0f));
    float dist = 1e38f;
    glm::vec3 nor;
    glm::vec2 scaling;

    float areaX = c.scale.y * c.scale.z;
    float areaY = c.scale.x * c.scale.z;
    float areaZ = c.scale.x * c.scale.y;

    float area = areaX + areaY + areaZ;
    if (r < areaX / area) {
        // x
        glm::vec3 p1 = glm::vec3(c.scale.x * 0.5f, c.scale.y * xi.x, c.scale.z * xi.y);
        if (dist > distance(p_local, p1)) {
            dist = distance(p_local, p1);
            hit_point = p1;
            nor = glm::vec3(1.0f, 0.0f, 0.0f);
            scaling = glm::vec2(c.scale.y, c.scale.z);
        }
        glm::vec3 p2 = glm::vec3(c.scale.x * -0.5f, c.scale.y * xi.x, c.scale.z * xi.y);
        if (dist > distance(p_local, p2)) {
            dist = distance(p_local, p2);
            hit_point = p2;
            nor = glm::vec3(-1.0f, 0.0f, 0.0f);
            scaling = glm::vec2(c.scale.y, c.scale.z);
        }
    }
    else if (r < (areaX + areaY) / area) {
        // y
        glm::vec3 p3 = glm::vec3(c.scale.x * xi.x, c.scale.y * 0.5f, c.scale.z * xi.y);
        if (dist > distance(p_local, p3)) {
            dist = distance(p_local, p3);
            hit_point = p3;
            nor = glm::vec3(0.0f, 1.0f, 0.0f);
            scaling = glm::vec2(c.scale.x, c.scale.z);
        }
        glm::vec3 p4 = glm::vec3(c.scale.x * xi.x, c.scale.y * -0.5f, c.scale.z * xi.y);
        if (dist > distance(p_local, p4)) {
            dist = distance(p_local, p4);
            hit_point = p4;
            nor = glm::vec3(0.0f, -1.0f, 0.0f);
            scaling = glm::vec2(c.scale.x, c.scale.z);
        }
    }
    else {
        // z
        glm::vec3 p5 = glm::vec3(c.scale.x * xi.x, c.scale.y * xi.y, c.scale.z * 0.5f);
        if (dist > distance(p_local, p5)) {
            dist = distance(p_local, p5);
            hit_point = p5;
            nor = glm::vec3(0.0f, 0.0f, 1.0f);
            scaling = glm::vec2(c.scale.y, c.scale.x);
        }
        glm::vec3 p6 = glm::vec3(c.scale.x * xi.x, c.scale.y * xi.y, c.scale.z * -0.5f);
        if (dist > distance(p_local, p6)) {
            dist = distance(p_local, p6);
            hit_point = p6;
            nor = glm::vec3(0.0f, 0.0f, -1.0f);
            scaling = glm::vec2(c.scale.y, c.scale.x);
        }
    }
    light.origin = isect_point;
    hit_point = multiplyMV(c.transform, glm::vec4(hit_point, 1.0f));
    light.direction = normalize(hit_point - isect_point);
    dist = distance(hit_point, isect_point);
    nor = glm::normalize(multiplyMV(c.transform, glm::vec4(nor, 0.0f)));
    float pdf = pow(dist, 2.0f) / (area);
    return pdf;

    // add intersection test
}

__host__ __device__ void barycentric(Triangle t, glm::vec3 p, glm::vec3& results) {
    glm::vec3 P1 = t.vertices[0].pos, P2 = t.vertices[1].pos, P3 = t.vertices[2].pos;
    glm::vec3 v1 = P2 - P1, v2 = P3 - P1, v3 = p - P1;
    float d11 = dot(v1, v1);
    float d12 = dot(v1, v2);
    float d22 = dot(v2, v2);
    float d31 = dot(v3, v1);
    float d32 = dot(v3, v2);
    float d = 1.0f / (d11 * d22 - d12 * d12);
    results[1] = (d22 * d31 - d12 * d32) * d;
    results[0] = (d11 * d32 - d12 * d31) * d;
    results[2] = 1.0f - results[0] - results[1];
}

__host__ __device__ float meshIntersectionTest(Geom m, Triangle* tris, glm::vec2& uv, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    Ray q;
    q.origin = multiplyMV(m.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(m.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmax = -1e38f, tmin = 1e38f;
    glm::vec3 tmin_n, tmax_n;
    size_t tri_hit = -1, tri_hit_hold = -1;

    for (size_t i = m.triIdx; i < m.triIdx + m.triCount; i++) {
        Triangle t = tris[i];

        glm::vec3 P1 = t.vertices[0].pos, P2 = t.vertices[1].pos, P3 = t.vertices[2].pos;
        
        glm::vec3 pos;
        if (glm::intersectRayTriangle(q.origin, q.direction, P1, P2, P3, pos)) {
            if (pos.z < tmin) {
                tmin = pos.z;
                tmin_n = t.vertices[0].nor;
                tri_hit = i;
            }
            if (pos.z > tmax) {
                tmax = pos.z;
                tmax_n = t.vertices[0].nor;
                tri_hit_hold = i;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
            tri_hit = tri_hit_hold;
        }

        // interpolate to find uv
        Triangle t = tris[tri_hit];
        glm::vec3 interp;
        barycentric(t, getPointOnRay(q, tmin), interp);

        glm::vec2 uv1 = t.vertices[0].uv, uv2 = t.vertices[1].uv, uv3 = t.vertices[2].uv;
        float x = interp[0] * uv1[0] + interp[1] * uv2[0] + interp[2] * uv3[0];
        float y = interp[0] * uv1[1] + interp[1] * uv2[1] + interp[2] * uv3[1];

        uv = interp[0] * uv1 + interp[1] * uv2 + interp[2] * uv3;

        intersectionPoint = multiplyMV(m.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));

        normal = glm::normalize(multiplyMV(m.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;


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
