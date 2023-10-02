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
    MaterialType judgeMaterialType(float reflectivity, float refractivity, float roughness);

    int loadGeom(string objectid);
    int loadCamera();

    void calculateAABB(Geom& geom);
    void buildKDTree();
    KDNode* build(std::vector<Geom>& geoms, int depth);
    int createKDAccelNodes(KDNode* node, int& index);

    bool loadObj(const Geom& geom, const string& objFile);
    
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> sortedGeoms;
    //std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<Light> lights;

    KDNode* kdRoot;
    int nodeCount = 0;
    std::vector<KDAccelNode> kdNodes;
    RenderState state;
};
