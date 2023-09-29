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

__host__ __device__ void get_sphere_uv(const glm::vec3& p, float& u, float& v) {
	// p: a given point on the sphere of radius one, centered at the origin.
	// u: returned value [0,1] of angle around the Y axis from X=-1.
	// v: returned value [0,1] of angle from Y=-1 to Y=+1.
	//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
	//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
	//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

	auto theta = std::acos(-p.y);
	auto phi = atan2(-p.z, p.x) + glm::pi<float>();

	u = phi / (2 * glm::pi<float>());
	v = theta / glm::pi<float>();
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
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, float& u, float& v) {
    float radius = 0.5f;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    //float vDotDirection = glm::dot(rt.origin, rt.direction);
    //float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2.0f));
    //if (radicand < 0) {
    //    return -1.0f;
    //}

    //float squareRoot = std::sqrtf(radicand);
    //float firstTerm = -vDotDirection;
    //float t1 = firstTerm + squareRoot;
    //float t2 = firstTerm - squareRoot;

    //float t = 0;
    //if (t1 < 0 && t2 < 0) {
    //    return -1;
    //} else if (t1 > 0 && t2 > 0) {
    //    t = min(t1, t2);
    //    outside = true;
    //} else {
    //    t = max(t1, t2);
    //    outside = false;
    //}

	glm::vec3 oc = ro;
	float a = glm::dot(rd, rd);
	float halfB = glm::dot(oc, rd);
	float c = glm::dot(oc, oc) - radius * radius;

	float discriminant = halfB * halfB - a * c;

	if (discriminant < 0.0f)
	{
		return -1;
	}

	float sqrtd = std::sqrtf(discriminant);
	float root = (-halfB - sqrtd) / a;

	if (root < 0.001f || 10000.0f < root)
	{
		root = (-halfB + sqrtd) / a;
		if (root < 0.001f || 10000.0f < root)
		{
			return -1;
		}
	}

    float t = root;

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

	outside = glm::dot(rd, normal) < 0.0f;

	normal = outside ? normal : -normal;

    get_sphere_uv(normal, u, v);

	//if (!outside) {
	//	normal = -normal;
	//}

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(Geom mesh, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	//mesh.triangle.v0 = { -1.0f, 0.0f, 0.0f };
	//mesh.triangle.v1 = {  1.0f, 0.0f, 0.0f };
	//mesh.triangle.v2 = {  0.0f, 1.0f, 0.0f };

    // E1
    glm::vec3 E1 = mesh.triangle.v1 - mesh.triangle.v0;

    // E2
    glm::vec3 E2 = mesh.triangle.v2 - mesh.triangle.v0;

    // P
    glm::vec3 P = glm::cross(rt.direction, E2);

    // determinant
    float det = dot(E1, P);

    // keep det > 0, modify T accordingly
    glm::vec3 T;
    if (det > 0)
    {
        T = rt.origin - mesh.triangle.v0;
    }
    else
    {
        T = mesh.triangle.v0 - rt.origin;
        det = -det;
    }

    // If determinant is near zero, ray lies in plane of triangle
    if (det < 0.0001f)
    {
        return -1.0f;
    }

    // Calculate u and make sure u <= 1
    float u = dot(T, P);

    if (u < 0.0f || u > det)
    {
        return -1.0f;
    }

    // Q
    glm::vec3 Q = glm::cross(T, E1);

    // Calculate v and make sure u + v <= 1
    float v = glm::dot(rt.direction, Q);

    if (v < 0.0f || u + v > det)
    {
        return -1.0f;
    }

    // Calculate t, scale parameters, ray intersects triangle
    float t = dot(E2, Q);

    float fInvDet = 1.0f / det;

    t *= fInvDet;
    u *= fInvDet;
    v *= fInvDet;

    glm::vec3 objspaceIntersection = (1.0f - u - v) * mesh.triangle.v0 + u * mesh.triangle.v1 + v * mesh.triangle.v2;

	intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
	normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normalize(cross(E1, E2)), 0.0f)));

	return glm::length(r.origin - intersectionPoint);
}

__device__ inline glm::vec2 sampleHDRMap(glm::vec3 v)
{
	const glm::vec2 invAtan = glm::vec2(0.1591f, 0.3183f);
	glm::vec2 uv = glm::vec2(atan2(v.z, v.x), asin(v.y));
	uv *= invAtan;
	uv += 0.5f;
	return uv;
}

