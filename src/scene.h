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
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	void loadTexture(const std::string& path, cudaTextureObject_t* cudaTextureObject, int type);
public:
    Scene(string filename);
    ~Scene();

    void loadTextures();

    void randomScene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
	std::vector<cudaArray*> textureData;
    std::vector<std::pair<std::string, int> > texturesMap;
    cudaTextureObject_t skyboxTextureObject = 0;
};
