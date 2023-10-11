#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <vector>
#include "textureStruct.h"

#define ONE_OVER_255 0.00392156862745098f

__device__ glm::vec4 sampleTextureRGBA(const Texture & texture, const glm::vec2 & uv) {
	assert(texture.nrChannels >= 4);
	int x = uv.x * (texture.width-1);
	int y = uv.y * (texture.height-1);
	int index = (x + (y * texture.width)) * texture.nrChannels;
	return glm::vec4(texture.data[index], texture.data[index + 1], texture.data[index + 2], texture.data[index+3]) * ONE_OVER_255;
};

__device__ glm::vec3 sampleTextureRGB(const Texture & texture, const glm::vec2 & uv) {
	assert(texture.nrChannels >= 3);
	int x = uv.x * (texture.width - 1);
	int y = uv.y * (texture.height - 1);
	int index = (x + (y * texture.width)) * texture.nrChannels;
	return glm::vec3(texture.data[index], texture.data[index + 1], texture.data[index + 2]) * ONE_OVER_255;
};

__device__ glm::vec2 sampleTextureRG(const Texture& texture, const glm::vec2& uv) {
	assert(texture.nrChannels >= 2);
	int x = uv.x * (texture.width - 1);
	int y = uv.y * (texture.height - 1);
	int index = (x + (y * texture.width)) * texture.nrChannels;
	return glm::vec2(texture.data[index] * ONE_OVER_255, texture.data[index + 1] * ONE_OVER_255);
};