#pragma once

#include "sceneStructs.h"
#include "utilities.h"

// ref: https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering

__host__ __device__ float sampleDistance(const Medium& medium, float tFar, float& weight, float& pdf,
	thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

	if (medium.mediumType == MediumType::NONSCATTERING) {
		pdf = 1.0f;
		return tFar;
	}
	else {
		float distance = -1.0 / medium.scatteringCoefficient;
		if (distance >= tFar) {
			pdf = 1.0f;
			return tFar;
		}

		pdf = exp(-medium.scatteringCoefficient * distance);
		return distance;
	}
}

__host__ __device__ glm::vec3 sampleScatterDirection(const Medium& medium, const glm::vec3& woW, float& pdf, thrust::default_random_engine& rng) {
	switch (medium.mediumType) {
		case MediumType::NONSCATTERING:
			pdf = 1.0f;
			return woW;
		case MediumType::ISOTROPIC:
			pdf = 1.0f / (4.0f * PI);
			// uniform sample sphere
			thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
			glm::vec2 sample(u01(rng), u01(rng));
			float z = 1.0f - 2.0f * sample.x;

			float theta = TWO_PI * sample.y;
			float r = sqrt(glm::max(0.0f, 1.0f - z * z));

			return glm::vec3(cos(theta) * r, sin(theta) * r, z);
	}
}

__host__ __device__ glm::vec3 calculateTransmission(const Medium& medium, float distance) {
	switch (medium.mediumType) {
		case MediumType::NONSCATTERING:
			return exp(-medium.absorptionCoefficient * distance);
		case MediumType::ISOTROPIC:
			return exp(-medium.absorptionCoefficient * distance);
	}
}

__host__ __device__ glm::vec3 sampleMediumTransmission(const Medium& medium, const float t,
	const float tFar, float& weight, float& distance,
	thrust::default_random_engine& rng) {
	float pdf = 1.0f;

	distance = sampleDistance(medium, tFar, weight, pdf, rng);
	return calculateTransmission(medium, glm::min(distance, t));
}