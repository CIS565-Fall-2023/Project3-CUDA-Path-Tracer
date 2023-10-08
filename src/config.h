#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "textureStruct.h"
#include <thrust/device_vector.h>

using namespace std;
/*
    As the poor support of blender over gltf, this raytracer adopts a mixed scene representation
    gltf is used to represent the scene, but some other information is still defined in the config file
    Practically, Camera, Environment Map, and some other parameters are defined in the config file
*/

using EnvrionmentalMapInfo = TextureInfo;

class SceneConfig {
private:
    ifstream fp_in;
    int loadCamera();
    int loadEnvironmentMap();
public:
    SceneConfig() = default;
    SceneConfig(string filename);

    bool has_env_map = false;
    EnvrionmentalMapInfo env_map;
    RenderState state;
    int geoms_size;
};

