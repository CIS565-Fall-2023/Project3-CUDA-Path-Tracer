#pragma once

#include <vector>
#include "scene.h"

#include <cuda_runtime.h>
#include <thrust/random.h>

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void showGBuffer(uchar4* pbo, bool ui_ShowGbuffer, bool ui_ShowNormal, bool ui_ShowPosition);
void showImage(uchar4* pbo, int iter);
void denoise(uchar4* pbo, int iter, float colorWeight, float normalWeight, float positionWeight, float filterSize);
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);