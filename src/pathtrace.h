#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtraceSortMat(uchar4 *pbo, int frame, int iteration);
void pathtraceSortMatWCache(uchar4 *pbo, int frame, int iteration);
void pathtraceSortMatWCacheBVH(uchar4 *pbo, int frame, int iteration,bool Cache, bool Antialiasing, bool sortMat, bool BVH,int shading);
