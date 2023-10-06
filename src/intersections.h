#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

__device__ const int MAX_STEPS = 100;
__device__ const float MAX_DIST = 100.0;
__device__ const float SURF_DIST = 0.01;

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

	//return glm::length(r.origin - intersectionPoint);
    return t;
}


__device__ float sdfCapsule(glm::vec3 p, glm::vec3 a, glm::vec3 b, float radius)
{
    glm::vec3 ab = b - a;
    glm::vec3 ap = p - a;

	float t = glm::dot(ab, ap) / glm::dot(ab, ab);
	t = glm::clamp(t, 0.0f, 1.0f);

    glm::vec3 c = a + t * ab;

	float d = length(p - c) - radius;

	return d;
}

__device__ float sdfTorus(glm::vec3 p, glm::vec2 r)
{
    float x = glm::length(glm::vec2(p.x, p.z)) - r.x;
	return glm::length(glm::vec2(x, p.y)) - r.y;
}

__device__ float sdfCylinder(glm::vec3 p, glm::vec3 a, glm::vec3 b, float radius)
{
    glm::vec3 ab = b - a;
    glm::vec3 ap = p - a;

	float t = glm::dot(ap, ab) / glm::dot(ab, ab);

    glm::vec3 c = a + t * ab;

	float x = glm::length(p - c) - radius;
	float y = (glm::abs(t - 0.5f) - 0.5f) * glm::length(ab);

	float e = glm::length(max(glm::vec2(x, y), 0.0f));
	float i = glm::min(glm::max(x, y), 0.0f);

	return e + i;
}

__device__ float sdfBox(glm::vec3 p, glm::vec3 size)
{
	p = glm::abs(p) - size;
	return glm::length(glm::max(p, 0.0f)) + glm::min(max(p.x, max(p.y, p.z)), 0.0f);
}

__device__ float getPlaneDist(glm::vec3 p)
{
	float d = p.y + 0.7f;
    return d;
}

__device__ float getSphereDist(glm::vec3 p, glm::vec4 s)
{

	glm::vec3 center = { s.x, s.y, s.z };

	float d = glm::length(p - center) - s.w;
    
    return d;
}

__device__ float getCapsuleDist(glm::vec3 p)
{
	float d = sdfCapsule(p, glm::vec3(-0.2f, -0.25f, 1.5f), glm::vec3(-0.2f, 0.25f, 1.5f), 0.25f);

    return d;
}

__device__ float displacement(glm::vec3 p)
{
    float freq = 15.0f;
    float amplitude = 0.1f;
    return amplitude * (sin(freq * p.x) * sin(freq * p.y) * sin(freq * p.z));
}

__device__ float getTorusDist(glm::vec3 p)
{
    float d1 = sdfTorus(p - glm::vec3(1.5, 0.0, 1.5), glm::vec2(0.3f, 0.15f));
    float d2 = displacement(p);
    return d1 + d2;
}

__device__ float getCylinderDist(glm::vec3 p)
{
	float d = sdfCylinder(p, glm::vec3(0.50f, -0.5f, 1.5f), glm::vec3(0.5f, 0.0f, 1.5f), 0.25f);
    return d;
}

__device__ float getCubeDist(glm::vec3 p)
{
    const float k = 2.0;
    float c = cos(k * p.x);
    float s = sin(k * p.x);
    glm::mat2  m = glm::mat2(c, -s, s, c);

    glm::vec2 rotated = m * glm::vec2(p.x, p.y);

    glm::vec3  q = glm::vec3(rotated.x, rotated.y, p.z);

    float d1 = sdfBox(q, glm::vec3(0.5f));
    float d2 = displacement(p);
    return d1+d2;
}

__device__ float opSmoothUnion(float d1, float d2, float k) 
{
    float h = glm::clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return glm::mix(d2, d1, h) - k * h * (1.0 - h);
}

__device__ float opSmoothSubtraction(float d1, float d2, float k) 
{
    float h = glm::clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return glm::mix(d2, -d1, h) + k * h * (1.0 - h);
}

__device__ float opSmoothIntersection(float d1, float d2, float k) 
{
    float h = glm::clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return glm::mix(d2, d1, h) + k * h * (1.0 - h);
}





__device__ float getDist(glm::vec3 p, ProceduralType type)
{
    float d = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    glm::vec4 s = glm::vec4(0.0f, 0.0f, 0.0f, 0.5);
    glm::vec4 s1 = glm::vec4(0.0f, 1.0f, 0.0f, 0.5);
    glm::vec4 s2 = glm::vec4(0.0f, 0.5f, 0.0f, 0.5);
    glm::vec4 s3 = glm::vec4(0.0f, 1.0f, 0.0f, 0.5);

    switch (type)
    {
    case ProceduralType::Plane:
		d = getPlaneDist(p);
        break;
    case ProceduralType::Cube:
        d = getCubeDist(p);
        break;
    case ProceduralType::Sphere:
		d = getSphereDist(p, s);
        break;
    case ProceduralType::Cylinder:
		d = getCylinderDist(p);
        break;
    case ProceduralType::Capsule:
		d = getCapsuleDist(p);
        break;
    case ProceduralType::Torus:
        d = getTorusDist(p);
        break;
    case ProceduralType::opSmoothUnion:
        d1 = getCubeDist(p);
        d2 = getSphereDist(p, s1);
        d = opSmoothUnion(d1, d2, 0.4);
        break;
    case ProceduralType::opSmoothSubtraction:
        d1 = getCubeDist(p);
        d2 = getSphereDist(p, s2);
        d = opSmoothSubtraction(d1, d2, 2);
        break;
    case ProceduralType::opSmoothIntersection:
        d1 = getCubeDist(p);
        d2 = getSphereDist(p, s3);
        d = opSmoothIntersection(d1, d2, 0.4);
        break;
    default:
        break;
    }

	return d;
}

__device__ float rayMarch(glm::vec3 ro, glm::vec3 rd, ProceduralType type)
{
	float dO = 0.0;

	for (int i = 0; i < MAX_STEPS; i++)
	{
		glm::vec3 p = ro + rd * dO;

		float dS = getDist(p, type);

		dO += dS;
		if (dO > MAX_DIST || dS < SURF_DIST)
		{
			break;
		}
	}

	return dO;
}

__device__ glm::vec3 getNormal(glm::vec3 p, ProceduralType type)
{
	glm::vec2 e = glm::vec2(0.001f, 0.0f);
	float d = getDist(p, type);
    glm::vec3 n = d - glm::vec3(getDist(p - glm::vec3(e.x, e.y, e.y), type),
		                         getDist(p - glm::vec3(e.y, e.x, e.y), type),
		                         getDist(p - glm::vec3(e.y, e.y, e.x), type));

	return normalize(n);
}

__device__ float proceduralIntersectionTest(Geom mesh, Ray r,
	glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float d = rayMarch(ro, rd, mesh.proceduralType);

    if (d < SURF_DIST || d > MAX_DIST)
    {
        return -1.0f;
    }

    glm::vec3 p = getPointOnRay(rt, d);

    glm::vec3 n = getNormal(p, mesh.proceduralType);

	intersectionPoint = multiplyMV(mesh.transform, glm::vec4(p, 1.0f));
	normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(n, 0.f)));

	outside = glm::dot(rd, normal) < 0.0f;

	normal = outside ? normal : -normal;

	d = glm::length(r.origin - intersectionPoint);

	return d;
}

__device__ inline glm::vec2 sampleHDRMap(glm::vec3 v)
{
	const glm::vec2 invAtan = glm::vec2(0.1591f, 0.3183f);
	glm::vec2 uv = glm::vec2(atan2(v.z, v.x), asin(v.y));
	uv *= invAtan;
	uv += 0.5f;
	return uv;
}
