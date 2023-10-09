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
    int loadObj(const char* filename);
    int loadGLTF(const char* filename);
    int loadTexture(string textureid);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Geom> lights;
    std::vector<Texture> textures;

    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;

    void constructBVHforAllGeom();
    int buildBVH(int start, int end);
    void updateBVHNode(const std::vector<BVHPrimitiveInfo>& primInfo, std::vector<BVHNode>& nodes, int idx);
    int subdivide(std::vector<BVHPrimitiveInfo>& primInfo, std::vector<BVHNode>& nodes, int idx, int& nodesVisited, int& maxSize);
};
