#pragma once

#include "lightStruct.h"
#include "sample.h"

__device__ glm::vec3 L(const Light & light, const glm::vec3 & p, const glm::vec3 & n, const glm::vec2 & uv, 
	const glm::vec3 & w /* Outgoing ray from light */
) {
	return light.scale * light.color;
	//return light.color;
}

__device__ LightLiSample sampleLi(const Light& light, const Triangle & tri, const ShadeableIntersection& intersection, const glm::vec2 & u) {
	auto ss = sampleTriangle(tri, u);
	if (ss.pdf == 0 || glm::length2(ss.intersection.intersectionPoint - intersection.intersectionPoint) < EPSILON) {
		return {};
	}

	glm::vec3 wi = glm::normalize(ss.intersection.intersectionPoint - intersection.intersectionPoint);
	if (glm::dot(wi, intersection.surfaceNormal) < 0  
		|| glm::dot(-wi, ss.intersection.surfaceNormal) < 0
		) 
	{
			return {};
	}
	ss.intersection.t = glm::length(ss.intersection.intersectionPoint - intersection.intersectionPoint);
	//float pdf = ss.pdf * glm::length2(ss.intersection.intersectionPoint - intersection.intersectionPoint) / glm::dot(ss.intersection.surfaceNormal, -wi);
//	printf("ss.pdf: %f distance_sqr: %f light_p: %f %f %f, inters: %f %f %f\n", 
//		ss.pdf, 
//		glm::length2(ss.intersection.intersectionPoint - intersection.intersectionPoint),
//		ss.intersection.intersectionPoint.x, ss.intersection.intersectionPoint.y, ss.intersection.intersectionPoint.z,
//intersection.intersectionPoint.x, intersection.intersectionPoint.y, intersection.intersectionPoint.z
//		);
	float pdf = ss.pdf * glm::length2(ss.intersection.intersectionPoint - intersection.intersectionPoint) / glm::dot(ss.intersection.surfaceNormal, -wi);
	//float pdf = ss.pdf / glm::dot(ss.intersection.surfaceNormal, -wi);
	auto Le = L(light, ss.intersection.intersectionPoint, ss.intersection.surfaceNormal, ss.intersection.uv, -wi);
	LightLiSample sample;
	sample.L = Le;
	sample.lightIntersection = ss.intersection;
	sample.wi = wi;
	sample.pdf = pdf;
	return sample;
}