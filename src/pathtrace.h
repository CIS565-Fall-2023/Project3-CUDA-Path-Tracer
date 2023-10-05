#pragma once

#include <vector>
#include "scene.h"

#include <cuda_runtime.h>
#include <thrust/random.h>

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);