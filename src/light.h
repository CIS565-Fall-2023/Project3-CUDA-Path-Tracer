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
	glm::vec3 surfaceToLight = ss.position - intersection.intersectionPoint;
	if (ss.pdf == 0 || glm::length2(surfaceToLight) < EPSILON) {
		return {};
	}

	glm::vec3 wi = glm::normalize(surfaceToLight);
	if (glm::dot(wi, intersection.surfaceNormal) < 0  
		|| glm::dot(-wi, ss.normal) < 0
		) 
	{
			return {};
	}
	float distance = glm::length(surfaceToLight);
	float pdf = ss.pdf * distance * distance/ glm::dot(ss.normal, -wi);
	auto Le = light.scale * light.color;
	return {Le, wi, ss.position, pdf, distance};
}