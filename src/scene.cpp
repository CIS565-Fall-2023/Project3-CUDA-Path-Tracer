#include "scene.h"
#include <iostream>
#include <cstring>
#include <filesystem>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

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

void ApplyTransform(int start_id, std::vector<glm::vec3>& vertices, int n_start_id, std::vector<glm::vec3>& normals,
                    const glm::vec3& translate, const glm::vec3& rotate, const glm::vec3& scale)
{
    glm::mat4 transform;
    glm::mat3 inv_transpose;
    ComputeTransform(translate, rotate, scale, transform, inv_transpose);
    for (int i = start_id; i < vertices.size(); ++i)
    {
        vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.f));
    }
    for (int i = n_start_id; i < normals.size(); ++i)
    {
        normals[i] = inv_transpose * normals[i];
    }
}

Scene::Scene(const std::filesystem::path& res_path, const std::string& scene_filename)
    : m_EnvMapTexObj(0)
{
    std::filesystem::path scene_path(res_path);
    scene_path.append(scene_filename);
    if (!std::filesystem::directory_entry(scene_path).exists())
    {
        printf("Error reading from %s - aborting!", scene_path.string().c_str());
        throw;
    }
    
    if (scene_path.extension().string() == ".json")
    {
        LoadSceneFromJSON(scene_path, res_path);
    }
}

void Scene::FreeScene()
{
    m_Materials.clear();
    m_Textures.clear();
    m_Vertices.clear();
    m_Normals.clear();
    m_UVs.clear();
    m_TriangleIdxs.clear();
    m_MaterialMap.clear();
    m_EnvMapTexObj = 0;
    state = RenderState();
}

void Scene::LoadSceneFromJSON(const std::filesystem::path& scene_path, const std::filesystem::path& res_path)
{
    FreeScene();

    std::cout << "Reading scene from Json: " << scene_path.string() << " ..." << std::endl;

    std::ifstream fp_in(scene_path);

    Json scene_json = Json::parse(fp_in);

    std::string res_path_str;
    SafeGet<std::string>(scene_json, "resources path", res_path_str);

    std::filesystem::path scene_res_path(res_path);

    if (scene_json.contains("camera")) LoadCameraFromJSON(scene_json["camera"]);
    if (scene_json.contains("materials")) LoadMaterialsFromJSON(scene_json["materials"], scene_res_path);
    if (scene_json.contains("geomerties")) LoadGeomsFromJSON(scene_json["geomerties"], scene_res_path);
    if (scene_json.contains("environment map")) LoadEnvironmentMapFromJSON(scene_json["environment map"], scene_res_path);

    std::cout << "Reading scene success!" << std::endl;
}

void Scene::LoadGeomsFromJSON(const Json& geometry_json, const std::filesystem::path& res_path)
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

        std::string obj_path_str;
        SafeGet<std::string>(geometry_json[i], "path", obj_path_str);

        std::filesystem::path obj_path(res_path);
        obj_path.append(obj_path_str);
        if (std::filesystem::directory_entry(obj_path).exists())
        {
            int v_start_id = m_Vertices.size();
            int n_start_id = m_Normals.size();
            std::cout << "Loading object: " << obj_path.string() << std::endl;
            ReadObj(obj_path.string(), material_id, res_path);
            ApplyTransform(v_start_id, m_Vertices, n_start_id, m_Normals, translate, rotate, scale);
            std::cout << "Finish loading object: " << obj_path.string() << std::endl;
        }
        else
        {
            std::cout << obj_path.string() << std::endl;
        }
    }
    std::cout << "Loading Geomerties Success!" << std::endl;
}

void Scene::LoadCameraFromJSON(const Json& camera_json) 
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

void Scene::LoadEnvironmentMapFromJSON(const Json& environment_json, const std::filesystem::path& res_path)
{
    std::cout << "Loading Environment Map ..." << std::endl;

    auto tex_obj = LoadTextureFromJSON(environment_json, res_path);
    if (tex_obj > 0)
    {
        m_EnvMapTexObj = tex_obj;
    }
    std::cout << "Loading Environment Map Success!" << std::endl;
}

void Scene::LoadMaterialsFromJSON(const Json& material_json, const std::filesystem::path& res_path)
{
    std::cout << "Loading Materials ..." << std::endl;
    for (unsigned int i = 0; i < material_json.size(); ++i)
    {
        Material material;
        std::string name, type;
        SafeGet<std::string>(material_json[i], "type", type);
        SafeGet<std::string>(material_json[i], "name", name);
        
        material.type = StringToMaterialType(type);

        m_MaterialMap.emplace(name, i);

        SafeGet<float>(material_json[i], "emittance", material.emittance);
        SafeGet<float>(material_json[i], "eta", material.eta);

        if(material_json[i].contains("albedo map"))
        {
            auto tex_obj = LoadTextureFromJSON(material_json[i]["albedo map"], res_path);
            if (tex_obj > 0)
            {
                material.type = static_cast<MaterialType>(material.type | MaterialType::Albedo_Texture);
                material.data.textures.albedo_tex.m_TexObj = tex_obj;
            }
        }
        else
        {
            SafeGetVec<glm::vec3, float, 3>(material_json[i], "albedo", material.data.values.albedo);
        }

        if (material_json[i].contains("normal map"))
        {
            auto tex_obj = LoadTextureFromJSON(material_json[i]["normal map"], res_path);
            if (tex_obj > 0)
            {
                material.type = static_cast<MaterialType>(material.type | MaterialType::Normal_Texture);
                material.data.textures.normal_tex.m_TexObj = tex_obj;
            }
        }

        m_Materials.push_back(std::move(material));
    }
    std::cout << "Loading Materials Success!" << std::endl;
}

void Scene::ReadObj(const std::string& obj_file_path,
                    unsigned int material_id, const std::filesystem::path& res_path)
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
    
    unsigned int v_start_id = m_Vertices.size();
    unsigned int n_start_id = m_Normals.size();
    unsigned int uv_start_id = m_UVs.size();

    m_Vertices.resize(m_Vertices.size() + attribs.vertices.size() / 3);
    m_Normals.resize(m_Normals.size() + attribs.normals.size() / 3);
    m_UVs.resize(m_UVs.size() + attribs.texcoords.size() / 2);

    std::memcpy(&m_Vertices[v_start_id],
                attribs.vertices.data(), 
                attribs.vertices.size() * sizeof(float));

    std::memcpy(&m_Normals[n_start_id],
                attribs.normals.data(),
                attribs.normals.size() * sizeof(float));

    std::memcpy(&m_UVs[uv_start_id],
                attribs.texcoords.data(),
                attribs.texcoords.size() * sizeof(float));

    int material_start_id = m_Materials.size();
    
    for (auto& material : reader.GetMaterials())
    {
        LoadMaterialsFromMTL(material, res_path);
    }

    for (auto& shape : shapes)
    {
        
        m_TriangleIdxs.reserve(m_TriangleIdxs.size() + shape.mesh.indices.size() / 3);
        for (int i = 0; i < shape.mesh.indices.size(); i += 3)
        {
            glm::ivec3 v_id{
                v_start_id + shape.mesh.indices[i].vertex_index,
                v_start_id + shape.mesh.indices[i + 1].vertex_index,
                v_start_id + shape.mesh.indices[i + 2].vertex_index
            };
            glm::ivec3 n_id{
                n_start_id + shape.mesh.indices[i].normal_index,
                n_start_id + shape.mesh.indices[i + 1].normal_index,
                n_start_id + shape.mesh.indices[i + 2].normal_index
            };
            glm::ivec3 uv_id{
                uv_start_id + shape.mesh.indices[i].texcoord_index,
                uv_start_id + shape.mesh.indices[i + 1].texcoord_index,
                uv_start_id + shape.mesh.indices[i + 2].texcoord_index
            };
            
            int mtl = (shape.mesh.material_ids[0] >= 0) ? shape.mesh.material_ids[0] + material_start_id : material_id;

            m_TriangleIdxs.emplace_back(v_id, n_id, uv_id, mtl);
        }
    }
}

void Scene::LoadMaterialsFromMTL(const MtlMaterial& mtl_material, const std::filesystem::path& res_path)
{
    Material material;
    ;

    material.type = MaterialType::MicrofacetMix;

    m_MaterialMap.emplace(mtl_material.name, m_Materials.size());

    if (mtl_material.diffuse_texname.size() > 0)
    {
        std::filesystem::path texture_path(res_path);
        texture_path.append(mtl_material.diffuse_texname);
        if (std::filesystem::directory_entry(texture_path).exists())
        {
            auto tex_obj = LoadTextureFromFile(texture_path.string(), true);
            material.type = static_cast<MaterialType>(material.type | MaterialType::Albedo_Texture);
            material.data.textures.albedo_tex.m_TexObj = tex_obj;
        }
    }
    else
    {
        material.data.values.albedo[0] = mtl_material.diffuse[0];
        material.data.values.albedo[1] = mtl_material.diffuse[1];
        material.data.values.albedo[2] = mtl_material.diffuse[2];
    }

    if (mtl_material.normal_texname.size() > 0)
    {
        std::filesystem::path texture_path(res_path);
        texture_path.append(mtl_material.normal_texname);
        if (std::filesystem::directory_entry(texture_path).exists())
        {
            auto tex_obj = LoadTextureFromFile(texture_path.string(), true);
            material.type = static_cast<MaterialType>(material.type | MaterialType::Normal_Texture);
            material.data.textures.normal_tex.m_TexObj = tex_obj;
        }
    }
    if (mtl_material.roughness_texname.size() > 0)
    {
        std::filesystem::path texture_path(res_path);
        texture_path.append(mtl_material.roughness_texname);
        if (std::filesystem::directory_entry(texture_path).exists())
        {
            auto tex_obj = LoadTextureFromFile(texture_path.string(), true);
            material.type = static_cast<MaterialType>(material.type | MaterialType::Roughness_Texture);
            material.data.textures.roughness_tex.m_TexObj = tex_obj;
        }
    }
    if (mtl_material.metallic_texname.size() > 0)
    {
        std::filesystem::path texture_path(res_path);
        texture_path.append(mtl_material.metallic_texname);
        if (std::filesystem::directory_entry(texture_path).exists())
        {
            auto tex_obj = LoadTextureFromFile(texture_path.string(), true);
            material.type = static_cast<MaterialType>(material.type | MaterialType::Metallic_Texture);
            material.data.textures.metallic_tex.m_TexObj = tex_obj;
        }
    }

    m_Materials.push_back(std::move(material));
}

cudaTextureObject_t Scene::LoadTextureFromJSON(const Json& texture_json, const std::filesystem::path& res_path)
{
    std::string tex_str;
    bool flip_v = false;
    SafeGet <std::string>(texture_json, "path", tex_str);
    SafeGet <bool>(texture_json, "flip", flip_v);
    std::filesystem::path texture_path(res_path);
    texture_path.append(tex_str);
    if (std::filesystem::directory_entry(texture_path).exists())
    {
        return LoadTextureFromFile(texture_path.string(), flip_v);
    }
    else
    {
        printf("Can not find texture with path: %s", texture_path.c_str());
        return 0;
    }
}

cudaTextureObject_t Scene::LoadTextureFromFile(const std::string& texture_str, bool flip_v)
{
    m_Textures.emplace_back(mkU<Texture2D>(texture_str, flip_v));
    return m_Textures.back()->m_TexObj;
}