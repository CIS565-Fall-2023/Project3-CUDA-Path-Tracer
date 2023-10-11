#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
void DrawGbuffer(int numIter);

//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
__host__ __device__ inline glm::vec3 util_postprocess_ACESFilm(glm::vec3 x)
{
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), glm::vec3(0), glm::vec3(1));
}

__host__ __device__ inline glm::vec3 util_postprocess_gamma(glm::vec3 x)
{
	return glm::pow(x, glm::vec3(1 / 2.2f));
}


__host__ __device__ inline int util_sample_alias_table(AliasBin* bins, int bSize, float random)
{
	float scaledRandom = random * bSize;
	int binIdx = min(scaledRandom, (float)bSize - 1);
	float u = scaledRandom - binIdx;
	if (u < bins[binIdx].q)
		return binIdx;
	else
		return bins[binIdx].alias;
}







