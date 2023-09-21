#pragma once

class CudaPathTracer;

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop(CudaPathTracer&);

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);