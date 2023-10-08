#pragma once

#include <vector>

class Scene;
class GuiDataContainer;

void InitDataContainer(GuiDataContainer* guiData, Scene* scene);
void UpdateDataContainer(GuiDataContainer* imGuiData, Scene* scene, float zoom, float theta, float phi);
void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(uchar4* pbo, int frame, int iteration);
