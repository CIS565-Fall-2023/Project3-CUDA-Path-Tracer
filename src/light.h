#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include <cuda_runtime.h>
#include "utilities.h"


class Light {

public:
	__device__ Light(const glm::vec3 & color, const float intensity):
		color(color), intensity(intensity){}
	__device__ ~Light() {};
	glm::vec3 color;
	float intensity;

	//__device__ virtual glm::vec3 get_emission(glm::vec3 r) const { return glm::vec3(); };
	//__device__ virtual glm::vec3 get_sample(glm::vec3 r) const {return glm::vec3();};
	__device__ virtual glm::vec3 get_emission(glm::vec3 r) const = 0;
	__device__ virtual glm::vec3 get_sample(glm::vec3 r) const = 0;
	
};

class AreaLight : public Light {
public:
	/* TODO: Add a rectangle here after adding rectangle in the primitive/geom part */
	__device__ AreaLight(const glm::vec3& color, const float intensity):
		Light(color, intensity) {}
	__device__ virtual ~AreaLight() {};
	__device__ glm::vec3 get_emission(glm::vec3 r) const override {
		// TODO
		return intensity * color;
	}
	__device__ glm::vec3 get_sample(glm::vec3 r) const override {
		// TODO
		return get_emission(r);
	}
};

//__device__ glm::vec3 AreaLight::get_emission(glm::vec3 r) const
//{
//	// TODO
//	return intensity * color;
//}
//
//__device__ glm::vec3 AreaLight::get_sample(glm::vec3 r) const
//{
//	// TODO
//	return get_emission(r);
//}