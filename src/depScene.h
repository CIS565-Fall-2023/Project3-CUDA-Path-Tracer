#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "light.h"
#include <thrust/device_vector.h>

using namespace std;

class HostScene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

    void initLightFromObject();
public:
    HostScene(string filename);
    ~HostScene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<std::shared_ptr<Light>> lights;
    RenderState state;
    int geoms_size;
};

