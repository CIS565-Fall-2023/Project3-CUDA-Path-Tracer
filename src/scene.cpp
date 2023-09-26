#include "scene.h"
#include <iostream>
#include <cstring>
#include <filesystem>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

typedef tinyobj::ObjReader ObjReader;
typedef tinyobj::ObjReaderConfig ObjReaderConfig;

template<typename T>
inline void SafeGet(const Json& json, const char* attr, T& value)
{
    if (json.contains(attr)) value = json[attr].get<T>();
}
template<typename T, typename D, unsigned int N>
inline void SafeGetVec(const Json& json, const char* attr, T& value)
{
    if (json.contains(attr))
    {
        for (int i = 0; i < N; ++i)
        {
            value[i] = json[attr][i].get<D>();
        }
    }
}

void ComputeTransform(const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale,
    glm::mat4& transform, glm::mat3& TransposeInvTransform)
{
    glm::mat4 T = glm::translate(glm::mat4(1.f), translate);

    glm::mat4 S = glm::scale(glm::mat4(1.f), scale);

    glm::mat4 Rx = glm::rotate(glm::mat4(1.f), glm::radians(rotate.x), { 1.f, 0.f, 0.f });
    glm::mat4 Ry = glm::rotate(glm::mat4(1.f), glm::radians(rotate.y), { 0.f, 1.f, 0.f });
    glm::mat4 Rz = glm::rotate(glm::mat4(1.f), glm::radians(rotate.z), { 0.f, 0.f, 1.f });

    transform = T * Rx * Ry * Rz * S;
    TransposeInvTransform = glm::transpose(glm::inverse(glm::mat3(transform)));
}

void AddPlane_Triangles(int start_id, std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles, int material_id)
{
    vertices.emplace_back(-1, -1, 0);
    vertices.emplace_back(1, -1, 0);
    vertices.emplace_back(1, 1, 0);
    vertices.emplace_back(-1, 1, 0);

    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 1, 2, material_id)); // front
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 2, 3, material_id)); // front
}

void AddCude_Triangles(int start_id, std::vector<glm::vec3>& vertices, std::vector<glm::ivec4>& triangles, int material_id)
{
    vertices.emplace_back(1, 1, -1);
    vertices.emplace_back(1, -1, -1);
    vertices.emplace_back(-1, -1, -1);
    vertices.emplace_back(-1, 1, -1);

    vertices.emplace_back(1, 1, 1);
    vertices.emplace_back(1, -1, 1);
    vertices.emplace_back(-1, -1, 1);
    vertices.emplace_back(-1, 1, 1);

    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 1, 2, material_id)); // front
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 2, 3, material_id)); // front
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(5, 4, 7, material_id)); // back
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(5, 7, 6, material_id)); // back
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(6, 7, 3, material_id)); // right
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(6, 3, 2, material_id)); // right
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 5, 1, material_id)); // left
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(0, 4, 5, material_id)); // left
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(3, 7, 4, material_id)); // top
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(3, 4, 0, material_id)); // top
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(2, 1, 5, material_id)); // bottom
    triangles.emplace_back(glm::ivec4(start_id, start_id, start_id, 0) + glm::ivec4(2, 5, 6, material_id)); // bottom
}

void ApplyTransform(int start_id, std::vector<glm::vec3>& vertices,
    const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale)
{
    glm::mat4 transform;
    glm::mat3 inv_transpose;
    ComputeTransform(translate, rotate, scale, transform, inv_transpose);
    for (int i = start_id; i < vertices.size(); ++i)
    {
        vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
    }
}

Scene::Scene(const std::filesystem::path& res_path, const std::string& scene_filename) 
{
    std::filesystem::path scene_path(res_path);
    scene_path.append(scene_filename);
    if (!std::filesystem::directory_entry(scene_path).exists())
    {
        printf("Error reading from %s - aborting!", scene_path.string().c_str());
        throw;
    }
    
    std::cout << "Reading scene from " << scene_path.string() << " ..." << std::endl;

    std::ifstream fp_in(scene_path);

    Json scene_json = Json::parse(fp_in);

    std::string res_path_str;
    SafeGet<std::string>(scene_json, "resources path", res_path_str);

    std::filesystem::path scene_res_path(res_path);

    LoadCamera(scene_json["camera"]);
    LoadMaterials(scene_json["materials"], scene_res_path); 
    LoadGeoms(scene_json["geomerties"], scene_res_path);

    std::cout << "Reading scene success!"<< std::endl;
}

void Scene::LoadGeoms(const Json& geometry_json, const std::filesystem::path& res_path)
{
    std::cout << "Loading Geometry ..." << std::endl;
    for (unsigned int i = 0; i < geometry_json.size(); ++i)
    {
        std::string name;
        std::string type;
        std::string material_name;

        glm::vec3 translate(0.f), rotate(0.f), scale(1.f);

        SafeGet<std::string>(geometry_json[i], "type", type);
        SafeGet<std::string>(geometry_json[i], "name", name);
        SafeGet<std::string>(geometry_json[i], "material", material_name);

        SafeGetVec<glm::vec3, float, 3>(geometry_json[i], "translation", translate);
        SafeGetVec<glm::vec3, float, 3>(geometry_json[i], "rotation", rotate);
        SafeGetVec<glm::vec3, float, 3>(geometry_json[i], "scale", scale);

        auto it = m_MaterialMap.find(material_name);
        int material_id = (it != m_MaterialMap.end() ? it->second: 0);

        if (type == "plane")
        {
            int start_id = m_Vertices.size();
            AddPlane_Triangles(start_id, m_Vertices, m_vIds, material_id);
            ApplyTransform(start_id, m_Vertices, translate, rotate, scale);
        }
        else if (type == "cube")
        {
            int start_id = m_Vertices.size();
            AddCude_Triangles(start_id, m_Vertices, m_vIds, material_id);
            ApplyTransform(start_id, m_Vertices, translate, rotate, scale);
        }
        else if (type == "obj")
        {
            std::string obj_path_str;
            SafeGet<std::string>(geometry_json[i], "path", obj_path_str);

            std::filesystem::path obj_path(res_path);
            obj_path.append(obj_path_str);
            if (std::filesystem::directory_entry(obj_path).exists())
            {
                int start_id = m_Vertices.size();
                ReadObj(obj_path.string(), material_id);
                ApplyTransform(start_id, m_Vertices, translate, rotate, scale);
            }
            else
            {
                std::cout << obj_path.string() << std::endl;
            }
        }
        else
        {
            assert(false);
        }
    }
    std::cout << "Loading Geomerties Success!" << std::endl;
}

void Scene::LoadCamera(const Json& camera_json) 
{
    std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;

    SafeGet<float>(camera_json, "fovy", camera.fovy);
    SafeGet<unsigned int>(camera_json, "interation", state.iterations);
    SafeGet<int>(camera_json, "depth", state.traceDepth);
    SafeGetVec<glm::vec3, float, 3>(camera_json, "ref", camera.ref);
    SafeGetVec<glm::vec3, float, 3>(camera_json, "position", camera.position);
    SafeGetVec<glm::vec3, float, 3>(camera_json, "up", camera.up);
    SafeGetVec<glm::ivec2, unsigned int, 2>(camera_json, "resolution", camera.resolution);

    std::cout << "Loading Camera Success!" << std::endl;
}

void Scene::LoadMaterials(const Json& material_json, const std::filesystem::path& res_path)
{
    std::cout << "Loading Materials ..." << std::endl;
    for (unsigned int i = 0; i < material_json.size(); ++i)
    {
        Material material;
        std::string name;
        SafeGet<std::string>(material_json[i], "name", name);

        m_MaterialMap.emplace(name, i);

        SafeGetVec<glm::vec3, float, 3>(material_json[i], "albedo", material.albedo);
        SafeGet<float>(material_json[i], "emittance", material.emittance);
        materials.push_back(std::move(material));
    }
    std::cout << "Loading Materials Success!" << std::endl;
}

void Scene::ReadObj(const std::string& obj_file_path,
                    unsigned int matrial_id)
{
    ObjReader reader;
    ObjReaderConfig config;
    
    // we want the mesh to be triangulated
    config.triangulate = true;
    //config.triangulation_method = "earcut";
    reader.ParseFromFile(obj_file_path, config);

    if (!reader.Error().empty()) std::cerr << "TinyObjReader: " << reader.Error();

    if (!reader.Warning().empty()) std::cout << "TinyObjReader: " << reader.Warning();

    auto& attribs = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    
    unsigned int start_id = m_Vertices.size();

    m_Vertices.resize(m_Vertices.size() + attribs.vertices.size() / 3);

    std::memcpy(&m_Vertices[start_id],
                attribs.vertices.data(), 
                attribs.vertices.size() * sizeof(float));

    for (auto& shape : shapes)
    {
        m_vIds.reserve(m_vIds.size() + shape.mesh.indices.size() / 3);
        for (int i = 0; i < shape.mesh.indices.size(); i += 3)
        {
            m_vIds.emplace_back(start_id + shape.mesh.indices[i].vertex_index,
                                   start_id + shape.mesh.indices[i + 1].vertex_index,
                                   start_id + shape.mesh.indices[i + 2].vertex_index,
                                   matrial_id);
        }
    }
}
