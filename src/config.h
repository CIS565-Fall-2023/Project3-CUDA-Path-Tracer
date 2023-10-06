#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <thrust/device_vector.h>

using namespace std;

class SceneConfig {
private:
    ifstream fp_in;
    int loadCamera();
public:
    SceneConfig(string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    int geoms_size;
};

