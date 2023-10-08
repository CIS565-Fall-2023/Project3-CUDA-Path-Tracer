#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

namespace Math {
	__host__ __device__ float luminance(const glm::vec3 & color) {
		return glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
	}

	/* Return a sampled barycentric coordinate for triangle */
	__host__ __device__ glm::vec2 uniformSampleTriangle(const glm::vec2& u) {
		float su0 = sqrtf(u.x);
		return glm::vec2(1 - su0, u.y * su0);
	}

}