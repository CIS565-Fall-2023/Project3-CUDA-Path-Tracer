#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
bool ValueChanged();
void ResetValueChanged();
void InitImguiData(GuiDataContainer* guiData);