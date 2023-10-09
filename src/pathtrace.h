#pragma once

#include <vector>
#include "scene.h"

namespace Pathtracer
{
    void InitDataContainer(GuiDataContainer* guiData);

    void init(Scene* scene);
    void initTextures();
    void initOIDN();
    void free();
    void freeTextures();
    void pathtrace(uchar4* pbo, int frame, int iteration);

    void onCamChanged();
}