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

typedef nlohmann::json Json;

namespace std
{
    namespace filesystem
    {
        class path;
    }
}

class Scene {
public:
    void LoadGeoms(const Json& geometry_json, const std::filesystem::path& res_path);
    void LoadMaterials(const Json& material_json, const std::filesystem::path& res_path);
    void LoadCamera(const Json& camera_json);

    void ReadObj(const std::string& obj_file_path,
                unsigned int matrial_id);
public:
    Scene(const std::filesystem::path& res_path, const std::string& scene_filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture2D> m_Textures;

    std::vector<glm::vec3> m_Vertices;
    std::vector<glm::vec3> m_Normals;
    std::vector<glm::vec2> m_UVs;
    std::vector<TriangleIdx> m_TriangleIdxs;
    
    std::unordered_map<std::string, unsigned int> m_MaterialMap;

    RenderState state;
};
