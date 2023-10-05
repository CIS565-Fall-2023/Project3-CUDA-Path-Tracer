#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"

struct ShapeSample {
	ShadeableIntersection intersection;
	float pdf;
};

/* Return a sampled barycentric coordinate for triangle */
__device__ static glm::vec2 uniformSampleTriangle(const glm::vec2 & u) {
	float su0 = sqrtf(u.x);
	return glm::vec2(1 - su0, u.y * su0);
}

__device__ static inline float sampleTrianglePdf(const Triangle& tri) {
	return 1.0f / tri.area();
}

__device__ static ShapeSample sampleTriangle(const Triangle & tri, const glm::vec2 & u){
	glm::vec2 b = uniformSampleTriangle(u);
	ShadeableIntersection intersection;
	intersection.surfaceNormal		= (1.0f - b.x - b.y) * tri.n1 + b.x * tri.n2 + b.y * tri.n3;
	intersection.intersectionPoint	= (1.0f - b.x - b.y) * tri.p1 + b.x * tri.p2 + b.y * tri.p3;
	intersection.uv					= (1.0f - b.x - b.y) * tri.uv1 + b.x * tri.uv2 + b.y * tri.uv3;
	intersection.bary				= b;
	return { intersection, sampleTrianglePdf(tri)};
}
