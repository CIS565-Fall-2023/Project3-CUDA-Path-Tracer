#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

namespace tinygltf {
    class Model;
    class Node;
}

class Scene {
private:
    static int id;
    struct Triangle {
        glm::vec3 v0, v1, v2;
        glm::vec3 normal0, normal1, normal2;
        glm::vec4 tangent0, tangent1, tangent2;
        glm::vec2 uv0, uv1, uv2;
        int id;
    };

    class Primitive {
    public:
        Primitive(const tinygltf::Primitive&, const Transformation& t, Scene*);
        DEFAULT_METHODS(Primitive)
    private:
        int materialid;
        std::vector<Triangle> tris;
        Scene* scene;
    };

    class Mesh
    {
    public:
        Mesh(const tinygltf::Node& node, const Transformation& transform, Scene*);
        DEFAULT_METHODS(Mesh)
    private:
        Transformation t;
        std::vector<Primitive> prims;
        Scene* scene;
    };
    static void loadExtensions(Material& material, const tinygltf::ExtensionMap& extensionMap);
    std::pair<PbrMetallicRoughness, Material::Type> loadPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbrMat);
    Camera& computeCameraParams(Camera& camera)const;
    int loadMaterial();
    bool loadTexture();
    int loadScene();
    void loadEnvMap();
    void loadSettings();
    void traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms);
    void loadNode(const tinygltf::Node& node);
    bool loadCamera(const tinygltf::Node&, const glm::mat4& transform);
    template<typename T>
    TextureInfo createTextureObj(int textureIndex, int width, int height, int component, const T* image, size_t size, int isSRGB = 0);
    void createCubemapTextureObj();
    std::vector<cudaArray_t> dev_tex_arrs_vec;
    std::vector<Mesh> meshes;
    tinygltf::Model* model;
    const int defaultMatId = 0;
public:
    struct Settings
    {
        const std::string filename = "Settings.json";
        std::string envMapFilename;
        std::string gltfPath;
        struct TransferableSettings {
            Material defaultMat;
            RenderState defaultRenderState;
            bool readFromFile;
            bool envMapEnabled;
            bool isProcedural;
            float scale;
            bool testNormal;
            bool testIntersect;
            glm::vec3 testColor;
        }trSettings;
        struct CameraSettings {
            bool dof;
            bool antiAliasing;
        }camSettings;
    } settings;
    struct Cubemap {
        cudaTextureObject_t texObj;
        cudaArray_t cubemapArray;
    }cubemap;
    Scene(std::string filename);
    ~Scene();
    TextureInfo envMapTexture;
    std::vector<cudaTextureObject_t> cuda_tex_vec;
    std::vector<TextureInfo> textures;
    std::vector<TriangleDetail> geoms;
    TBB tbb;
    TBVH tbvh;
    std::vector<Material> materials;
    std::vector<Camera> cameras;
    RenderState state;
};
