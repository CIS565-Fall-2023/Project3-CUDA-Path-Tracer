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
    int loadMesh(const string& fp, int& primStartIdx, int& primCnt);

    int buildBVH(int startPrim, int numPrim); // return BVHRoot Idx, take start Idx in Prim Buffer + Prims Cnt in Mesh
    void bvhUpdateBounds(BVHNode& node);
    float bvhBestSplitPlane(BVHNode& node, int& axis, float& splitPos, AABoundBox& leftChild, AABoundBox& rightChild);
    void bvhSubdivide(BVHNode& node);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<Triangle> prims;
    std::vector<BVHNode> BVHNodes;
    RenderState state;
};
