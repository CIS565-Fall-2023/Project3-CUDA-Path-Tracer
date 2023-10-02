#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    virtual int loadMaterial(string materialid);
    virtual int loadGeom(string objectid);
    virtual int loadCamera();
public:
    Scene() = default;
    Scene(string filename);
    virtual ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};

// class ObjScene : public Scene
// {
// private:
//     int loadMaterial(string materialid) override;
//     int loadGeom(string objectid) override;
//     int loadCamera() override;
// public:
//     ObjScene(string filename);
// };

// class GLTFScene : public Scene
// {
// private:
//     int loadMaterial(string materialid) override;
//     int loadGeom(string objectid) override;
//     int loadCamera() override;
// public:
//     GLTFScene(string filename);
// };
