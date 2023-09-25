#pragma once

#include <vector>
#include "scene.h"

static bool currentlyCaching = false;

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

void pathtraceFreeAll();
