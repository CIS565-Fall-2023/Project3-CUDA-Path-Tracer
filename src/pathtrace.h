#pragma once

#include <vector>
#include "config.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(SceneConfig *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtraceInitBeforeMainLoop(SceneConfig* config);
void pathtraceFreeAfterMainLoop();
