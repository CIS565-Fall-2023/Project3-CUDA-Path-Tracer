#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "hdrloader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    MaterialType judgeMaterialType(float reflectivity, float refractivity, float roughness);

    int loadGeom(string objectid);
    int loadCamera();

    void calculateAABB(Geom& geom);
    void buildKDTree();
    KDNode* build(std::vector<Geom>& geoms, int depth);
    int createKDAccelNodes(KDNode* node, int& index);

    bool loadObj(const Geom& geom, const string& objFile);

    bool loadHDR(const string& hdrFile);
    
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> sortedGeoms;
    std::vector<Material> materials;

    // lights
    std::vector<Light> lights;
    int numLights;

    // kd tree
    KDNode* kdRoot;
    int nodeCount = 0;
    std::vector<KDAccelNode> kdNodes;

    // hdr image
    HDRLoaderResult hdrResult;
    std::vector<glm::vec3> hdrImage;

    RenderState state;
};
