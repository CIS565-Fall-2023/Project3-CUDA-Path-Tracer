#pragma once

#include <vector>
#include "depScene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(HostScene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
