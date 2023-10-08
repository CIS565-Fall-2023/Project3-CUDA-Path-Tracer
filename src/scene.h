#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "cudaTexture.h"

#include <json.hpp>
#include <tiny_obj_loader.h>
typedef nlohmann::json Json;

#include <tiny_obj_loader.h>

typedef tinyobj::ObjReader ObjReader;
typedef tinyobj::ObjReaderConfig ObjReaderConfig;
typedef tinyobj::material_t MtlMaterial;

namespace std
{
    namespace filesystem
    {
        class path;
    }
}

class Scene {
public:
    void FreeScene();

    void LoadSceneFromJSON(const std::filesystem::path& scene_path, const std::filesystem::path& res_path);

    void LoadGeomsFromJSON(const Json& geometry_json, const std::filesystem::path& res_path);
    void LoadMaterialsFromJSON(const Json& material_json, const std::filesystem::path& res_path);
    void LoadCameraFromJSON(const Json& camera_json);
    void LoadEnvironmentMapFromJSON(const Json& environment_json, const std::filesystem::path& res_path);

    void ReadObj(const std::string& obj_file_path, unsigned int matrial_id, const std::filesystem::path& res_path);
    void LoadMaterialsFromMTL(const MtlMaterial& mtl_material, const std::filesystem::path& res_path);
    cudaTextureObject_t LoadTextureFromJSON(const Json& texture_json, const std::filesystem::path& res_path);
    cudaTextureObject_t LoadTextureFromFile(const std::string& texture_str, bool flip_v);

public:
    Scene(const std::filesystem::path& res_path, const std::string& scene_filename);
    ~Scene() = default;

    std::vector<Material> m_Materials;
    std::vector<uPtr<Texture2D>> m_Textures;

    std::vector<glm::vec3> m_Vertices;
    std::vector<glm::vec3> m_Normals;
    std::vector<glm::vec2> m_UVs;
    std::vector<TriangleIdx> m_TriangleIdxs;
    std::vector<glm::i16vec4> m_Indices;
    
    std::unordered_map<std::string, unsigned int> m_MaterialMap;
    
    cudaTextureObject_t m_EnvMapTexObj;
    RenderState state;
};
