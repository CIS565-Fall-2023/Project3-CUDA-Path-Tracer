#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include <cuda_runtime.h>
#include "utilities.h"
#include "sceneStructs.h"

enum LightType {
	POINT_LIGHT,
	AREA_LIGHT,
	SPOT_LIGHT,
	INNIFITE_AREA_LIGHT
};

struct Light {
	LightType type;
	int primIndex;

	/* Unimplemented for now */
	glm::mat4x4 w2l; // world to light matrix
	int nSample = 1;
	float scale;
	bool isDelta;
	glm::vec3 color;
	int textureID; // Used for envrionment map / infinite area light
};

struct LightLiSample {
	glm::vec3 L;
	glm::vec3 wi;
	float pdf;
	glm::vec3 intersectPoint;
};