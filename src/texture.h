#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <vector>
#include "textureStruct.h"

__device__ glm::vec4 sampleTextureRGBA(Texture texture, float u, float v) {
	int x = u * texture.width;
	int y = v * texture.height;
	int index = (x + (y * texture.width)) * texture.nrChannels;
	return glm::vec4(texture.data[index], texture.data[index + 1], texture.data[index + 2], texture.data[index+3]);
};

__device__ glm::vec3 sampleTextureRGB(Texture texture, float u, float v) {
	int x = u * texture.width;
	int y = v * texture.height;
	int index = (x + (y * texture.width)) * texture.nrChannels;
	return glm::vec3(texture.data[index], texture.data[index + 1], texture.data[index + 2]);
};