#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void updateBBox(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
void buildBVHTree(int startIndexBVH, int startIndexTri, GLTFMesh mesh1, int triCount);
