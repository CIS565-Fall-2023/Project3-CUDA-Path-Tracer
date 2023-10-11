#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include "mathUtil.h"
struct ShapeSample {
	glm::vec3 position;
	glm::vec3 normal;
	float pdf;
};

__device__ static inline float sampleTrianglePdf(const Triangle& tri) {
	return 1.0f / tri.area();
}

__device__ static ShapeSample sampleTriangle(const Triangle & tri, const glm::vec2 & u){
	glm::vec2 b = Math::uniformSampleTriangle(u);
	ShadeableIntersection intersection;
	intersection.surfaceNormal		= (1.0f - b.x - b.y) * tri.n1 + b.x * tri.n2 + b.y * tri.n3;
	intersection.intersectionPoint	= (1.0f - b.x - b.y) * tri.p1 + b.x * tri.p2 + b.y * tri.p3;
	return { 
		(1.0f - b.x - b.y) * tri.p1 + b.x * tri.p2 + b.y * tri.p3, // position 
		(1.0f - b.x - b.y) * tri.n1 + b.x * tri.n2 + b.y * tri.n3, // normal
		sampleTrianglePdf(tri)
	};
}
