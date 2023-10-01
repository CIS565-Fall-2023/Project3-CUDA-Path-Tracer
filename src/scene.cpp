#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>

#include "scene.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadObject(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "SKYBOX") == 0) {
                loadSkybox();
                cout << " " << endl;
            }
        }
    }
}

void Scene::loadSkybox()
{
    std::string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::cout << "Loading Skybox " << line << " ..." << endl;
        textureLoadJobs.emplace_back(line, -1);
    }
}

namespace std {
    template <>
    struct hash<std::pair<glm::vec3, glm::vec2>> {
        std::size_t operator()(const std::pair<glm::vec3, glm::vec2>& vertex) const {
            return ((hash<float>()(vertex.first.x) ^
                (hash<float>()(vertex.first.y) << 1)) >> 1) ^
                (hash<float>()(vertex.first.z) << 1) ^ (hash<float>()(vertex.second.x) << 2) ^ ((hash<float>()(vertex.second.y) << 2) >> 2);
        }
    };

}

void Scene::LoadAllTextures()
{
    for (auto& p : textureLoadJobs)
    {
        cudaTextureObject_t* texObj = p.second == -1 ? &skyboxTextureObj : &materials[p.second].baseColorMap;
        if (!strToTextureObj.count(p.first))
        {
            loadTextureFromFile(p.first, texObj, p.second != -1);
            strToTextureObj[p.first] = *texObj;
        }
        else
        {
            *texObj = strToTextureObj[p.first];
        }
    }

    for (auto& p : gltfTextureLoadJobs)
    {
        Material& mat = materials[p.matIndex];
        cudaTextureObject_t* texObj;
        switch (p.texType)
        {
        case TextureType::color:
            texObj = &mat.baseColorMap;
            break;
        case TextureType::normal:
            texObj = &mat.normalMap;
            break;
        case TextureType::metallicroughness:
            texObj = &mat.metallicRoughnessMap;
            break;
        }
        LoadTextureFromMemory(p.buffer, p.width, p.height, p.bits, p.component, texObj);
        delete[] p.buffer;
    }
}

Scene::~Scene()
{
    for (auto& p : strToTextureObj)
    {
        cudaDestroyTextureObject(materials[p.second].baseColorMap);
    }
    for (auto& p : textureDataPtrs)
    {
        cudaFreeArray(p);
    }
}

void Scene::LoadTextureFromMemory(void* data, int width, int height, int bits, int channels, cudaTextureObject_t* texObj)
{
    assert(channels == 4);
    cudaError_t err;
    size_t dataSize = width * height * 4 * (bits >> 3);
    cudaArray_t cuArray;
    cudaChannelFormatKind format = bits == 8 ? cudaChannelFormatKindUnsigned : cudaChannelFormatKindFloat;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bits, bits, bits, bits, format);
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    textureDataPtrs.emplace_back(cuArray);
    cudaMemcpyToArray(cuArray, 0, 0, data, width * height * 4 * (bits >> 3), cudaMemcpyHostToDevice);

    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    resDesc.res.linear.desc = cudaCreateChannelDesc(bits, bits, bits, bits, format);
    resDesc.res.linear.sizeInBytes = dataSize;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = bits == 8 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = 1;
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void Scene::loadTextureFromFile(const std::string& texturePath, cudaTextureObject_t* texObj, int type)
{
    int width, height, channels;
    if (!type)
    {
        unsigned char* data = stbi_load(texturePath.c_str(), &width, &height, &channels, 4);
        if (data) {
            LoadTextureFromMemory(data, width, height, 8, 4, texObj);
            stbi_image_free(data);
        }
        else {
            printf("Failed to load image: %s\n", stbi_failure_reason());
        }
    }
    else
    {
        float* data = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (data) {
            LoadTextureFromMemory(data, width, height, 32, 4, texObj);
            stbi_image_free(data);
        }
        else {
            printf("Failed to load image: %s\n", stbi_failure_reason());
        }
    }
}


//load model using tinyobjloader and tinygltf
bool Scene::loadModel(const string& modelPath, int objectid, bool useVertexNormal)
{
    cout << "Loading Model " << modelPath << " ..." << endl;
    if (modelPath.substr(modelPath.size() - 3, 3) == "obj")//load obj
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> aShapes;
        std::vector<tinyobj::material_t> aMaterials;
        std::string warn;
        std::string err;
        std::string mtlPath = modelPath.substr(0, modelPath.find_last_of('/') + 1);
        bool ret = tinyobj::LoadObj(&attrib, &aShapes, &aMaterials, &warn, &err, modelPath.c_str(), mtlPath.c_str());
        if (!warn.empty()) std::cout << warn << std::endl;

        if (!err.empty()) std::cerr << err << std::endl;

        if (!ret)  return false;

        int matOffset = materials.size();
        for (const auto& mat : aMaterials)
        {
            Material newMat{};
            newMat.color[0] = mat.diffuse[0];
            newMat.color[1] = mat.diffuse[1];
            newMat.color[2] = mat.diffuse[2];
            if (!mat.diffuse_texname.empty())
            {
                textureLoadJobs.emplace_back(mtlPath + mat.diffuse_texname, materials.size());
            }
            materials.emplace_back(newMat);

        }


        std::unordered_map<std::pair<glm::vec3, glm::vec2>, unsigned> vertex_set;
        for (const auto& shape : aShapes)
        {
            for (const auto& index : shape.mesh.indices)
            {
                glm::vec3 tmp_pos;
                tmp_pos.x = attrib.vertices[3 * index.vertex_index + 0];
                tmp_pos.y = attrib.vertices[3 * index.vertex_index + 1];
                tmp_pos.z = attrib.vertices[3 * index.vertex_index + 2];
                glm::vec2 tmp_uv;
                tmp_uv.x = index.texcoord_index >= 0 ? attrib.texcoords[2 * index.texcoord_index + 0] : -1.0;
                tmp_uv.y = index.texcoord_index >= 0 ? attrib.texcoords[2 * index.texcoord_index + 1] : -1.0;
                glm::vec3 tmp_normal;
                if (useVertexNormal)
                {
                    tmp_normal.x = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 0] : -1.0;
                    tmp_normal.y = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 1] : -1.0;
                    tmp_normal.z = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 2] : -1.0;
                }
                auto newVert = make_pair(tmp_pos, tmp_uv);
                if (!vertex_set.count(newVert))
                {
                    vertex_set[newVert] = verticies.size();
                    verticies.emplace_back(tmp_pos);
                    uvs.emplace_back(tmp_uv);
                    if (useVertexNormal) normals.emplace_back(tmp_normal);             
                }
            }
        }

        int modelStartIdx = objects.size();

        for (const auto& shape : aShapes) 
        {
            Object model;
            model.type = TRIANGLE_MESH;
            model.triangleStart = triangles.size();
            model.materialid = shape.mesh.material_ids[0] + matOffset;//Assume per mesh material
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) 
            {
                int fv = shape.mesh.num_face_vertices[f];
                assert(fv == 3);
                glm::ivec3 triangle;
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    glm::vec3 tmp;
                    tmp.x = attrib.vertices[3 * idx.vertex_index + 0];
                    tmp.y = attrib.vertices[3 * idx.vertex_index + 1];
                    tmp.z = attrib.vertices[3 * idx.vertex_index + 2];
                    glm::vec2 tmp_uv;
                    tmp_uv.x = idx.texcoord_index >= 0 ? attrib.texcoords[2 * idx.texcoord_index + 0] : -1.0;
                    tmp_uv.y = idx.texcoord_index >= 0 ? attrib.texcoords[2 * idx.texcoord_index + 1] : -1.0;
                    int vIdx = vertex_set[make_pair(tmp, tmp_uv)];
                    assert(vIdx >= 0 && vIdx < verticies.size());
                    triangle[v] = vIdx;
                }
                triangles.emplace_back(triangle);
                index_offset += fv;
            }
            model.triangleEnd = triangles.size();
            objects.emplace_back(model);
        }
        int modelEndIdx = objects.size();

        std::string line;
        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            int mID = atoi(tokens[1].c_str());
            if (mID != -1)
            {
                for (int i = modelStartIdx; i != modelEndIdx; i++)
                {
                    objects[i].materialid = mID;
                }
                cout << "Connecting Geom " << objectid << " to Material " << mID << "..." << endl;
            }
        }

        ObjectTransform modelTrans;
        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) 
        {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) 
            {
                modelTrans.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) 
            {
                modelTrans.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) 
            {
                modelTrans.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        modelTrans.transform = utilityCore::buildTransformationMatrix(
            modelTrans.translation, modelTrans.rotation, modelTrans.scale);
        modelTrans.inverseTransform = glm::inverse(modelTrans.transform);
        modelTrans.invTranspose = glm::inverseTranspose(modelTrans.transform);

        for (int i = modelStartIdx; i != modelEndIdx; i++)
        {
            objects[i].Transform = modelTrans;
        }

    }
    else//gltf
    {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, modelPath.c_str());

        if (!warn.empty())  std::cout << "Tiny GLTF Warn: " << warn << std::endl;

        if (!err.empty()) std::cout << "Tiny GLTF Err: " << err << std::endl;

        if (!ret) 
        {
            std::cout << "Failed to parse glTF" << std::endl;
            return -1;
        }
        int matOffset = materials.size();

        for (size_t i = 0; i < model.materials.size(); i++)
        {
            Material newMat;
            newMat.type = metallicWorkflow;
            auto& gltfMat = model.materials[i];
            auto& pbr = gltfMat.pbrMetallicRoughness;
            newMat.color[0] = pbr.baseColorFactor[0];
            newMat.color[1] = pbr.baseColorFactor[1];
            newMat.color[2] = pbr.baseColorFactor[2];
            newMat.roughness = pbr.roughnessFactor;
            newMat.metallic = pbr.metallicFactor;
            auto& baseColorTex = pbr.baseColorTexture;
            auto& metallicRoughnessTex = pbr.metallicRoughnessTexture;
            auto& normalTex = gltfMat.normalTexture;
            if (baseColorTex.index != -1)
            {
                assert(baseColorTex.texCoord == 0);//multi texcoord is not supported
                auto& tex = model.textures[baseColorTex.index];
                auto& image = model.images[tex.source];
                char* tmpBuffer = new char[image.image.size()];
                memcpy(tmpBuffer, &image.image[0], image.image.size());
                gltfTextureLoadJobs.emplace_back(tmpBuffer, materials.size(), TextureType::color, image.width, image.height, image.bits, image.component);
            }
            if (metallicRoughnessTex.index != -1)
            {
                assert(metallicRoughnessTex.texCoord == 0);//multi texcoord is not supported
                auto& tex = model.textures[metallicRoughnessTex.index];
                auto& image = model.images[tex.source];
                char* tmpBuffer = new char[image.image.size()];
                memcpy(tmpBuffer, &image.image[0], image.image.size());
                gltfTextureLoadJobs.emplace_back(tmpBuffer, materials.size(), TextureType::metallicroughness, image.width, image.height, image.bits, image.component);
            }
            if (normalTex.index != -1)
            {
                assert(normalTex.texCoord == 0);//multi texcoord is not supported
                auto& tex = model.textures[normalTex.index];
                auto& image = model.images[tex.source];
                char* tmpBuffer = new char[image.image.size()];
                memcpy(tmpBuffer, &image.image[0], image.image.size());
                gltfTextureLoadJobs.emplace_back(tmpBuffer, materials.size(), TextureType::normal, image.width, image.height, image.bits, image.component);
            }
            materials.emplace_back(newMat);
        }

        int triangleIdxOffset = verticies.size();//each gltf file assume a index starts with 0
        int modelStartIdx = objects.size();
        for (size_t i = 0; i < model.nodes.size(); ++i) 
        {
            tinygltf::Node& node = model.nodes[i];
            if (node.camera != -1 || node.mesh == -1) continue;//ignore GLTF's camera
            auto& mesh = model.meshes[node.mesh];
            for (auto& primtive : mesh.primitives)
            {
                assert(primtive.mode == TINYGLTF_MODE_TRIANGLES);
                Object newModel;
                newModel.type = TRIANGLE_MESH;
                newModel.triangleStart = triangles.size();
                newModel.materialid = primtive.material + matOffset;
                
                int indicesAccessorIdx = primtive.indices;
                int positionAccessorIdx = -1, normalAccessorIdx = -1, texcoordAccessorIdx = -1;
                if (primtive.attributes.count("POSITION"))
                {
                    positionAccessorIdx = primtive.attributes["POSITION"];
                }
                if (primtive.attributes.count("NORMAL")) 
                {
                    normalAccessorIdx = primtive.attributes["NORMAL"];
                }
                if (primtive.attributes.count("TEXCOORD_0"))
                {
                    texcoordAccessorIdx = primtive.attributes["TEXCOORD_0"];
                }
                assert(positionAccessorIdx != -1 && indicesAccessorIdx != -1);
                //Load indices
                auto& indicesAccessor = model.accessors[indicesAccessorIdx];
                if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT || indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    auto& bView = model.bufferViews[indicesAccessor.bufferView];
                    unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + indicesAccessor.byteOffset;
                    for (int i = 0; i < indicesAccessor.count; i+=3)
                    {
                        glm::ivec3 tri;
                        tri.x = *(unsigned short*)(ptr + sizeof(short) * (i + 0)) + triangleIdxOffset;
                        tri.y = *(unsigned short*)(ptr + sizeof(short) * (i + 1)) + triangleIdxOffset;
                        tri.z = *(unsigned short*)(ptr + sizeof(short) * (i + 2)) + triangleIdxOffset;
                        triangles.emplace_back(tri);
                    }
                }
                else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_INT || indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    auto& bView = model.bufferViews[indicesAccessor.bufferView];
                    unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + indicesAccessor.byteOffset;
                    for (int i = 0; i < indicesAccessor.count; i++)
                    {
                        glm::ivec3 tri;
                        tri.x = *(unsigned int*)(ptr + sizeof(int) * (i + 0)) + triangleIdxOffset;
                        tri.y = *(unsigned int*)(ptr + sizeof(int) * (i + 1)) + triangleIdxOffset;
                        tri.z = *(unsigned int*)(ptr + sizeof(int) * (i + 2)) + triangleIdxOffset;
                        triangles.emplace_back(tri);
                    }
                }
                else assert(0);//unexpected
                newModel.triangleEnd = triangles.size();
                objects.emplace_back(newModel);
                //Load position
                auto& positionAccessor = model.accessors[positionAccessorIdx];
                if (positionAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
                {
                    auto& bView = model.bufferViews[positionAccessor.bufferView];
                    unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + positionAccessor.byteOffset;
                    for (int i = 0; i < positionAccessor.count; i++)
                    {
                        glm::vec3 pos;
                        pos.x = *(float*)(ptr + sizeof(float) * (i * 3 + 0));
                        pos.y = *(float*)(ptr + sizeof(float) * (i * 3 + 1));
                        pos.z = *(float*)(ptr + sizeof(float) * (i * 3 + 2));
                        verticies.emplace_back(pos);
                    }
                }
                else assert(0);//unexpected
                //Load normals
                auto& normalAccessor = model.accessors[normalAccessorIdx];;
                if (useVertexNormal && normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
                {
                    auto& bView = model.bufferViews[normalAccessor.bufferView];
                    unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + normalAccessor.byteOffset;
                    for (int i = 0; i < normalAccessor.count; i++)
                    {
                        glm::vec3 normal;
                        normal.x = *(float*)(ptr + sizeof(float) * (i * 3 + 0));
                        normal.y = *(float*)(ptr + sizeof(float) * (i * 3 + 1));
                        normal.z = *(float*)(ptr + sizeof(float) * (i * 3 + 2));
                        normals.emplace_back(normal);
                    }
                }
                else assert(0);//unexpected
                if(useVertexNormal)
                    assert(verticies.size() == normals.size());
                //Load uv
                auto& texcoordAccessor = model.accessors[texcoordAccessorIdx];
                if (texcoordAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
                {
                    auto& bView = model.bufferViews[texcoordAccessor.bufferView];
                    unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + texcoordAccessor.byteOffset;
                    for (int i = 0; i < texcoordAccessor.count; i++)
                    {
                        glm::vec2 uv;
                        uv.x = *(float*)(ptr + sizeof(float) * (i * 2 + 0));
                        uv.y = *(float*)(ptr + sizeof(float) * (i * 2 + 1));
                        uvs.emplace_back(uv);
                    }
                }
                else assert(0);//unexpected
                assert(verticies.size() == uvs.size());

            }
        }
        int modelEndIdx = objects.size();

        std::string line;
        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            int mID = atoi(tokens[1].c_str());
            if (mID != -1)
            {
                for (int i = modelStartIdx; i != modelEndIdx; i++)
                {
                    objects[i].materialid = mID;
                }
                cout << "Connecting Geom " << objectid << " to Material " << mID << "..." << endl;
            }
        }

        ObjectTransform modelTrans;
        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0)
            {
                modelTrans.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0)
            {
                modelTrans.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0)
            {
                modelTrans.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        modelTrans.transform = utilityCore::buildTransformationMatrix(
            modelTrans.translation, modelTrans.rotation, modelTrans.scale);
        modelTrans.inverseTransform = glm::inverse(modelTrans.transform);
        modelTrans.invTranspose = glm::inverseTranspose(modelTrans.transform);

        for (int i = modelStartIdx; i != modelEndIdx; i++)
        {
            objects[i].Transform = modelTrans;
        }
    }

    return true;
}

bool Scene::loadGeometry(const string& type, int objectid)
{
    string line;
    Object newGeom;
    //load geometry type
    if (type == "sphere") {
        std::cout << "Creating new sphere..." << endl;
        newGeom.type = SPHERE;
    }
    else if (type == "cube") {
        std::cout << "Creating new cube..." << endl;
        newGeom.type = CUBE;
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        newGeom.materialid = atoi(tokens[1].c_str());
        std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            newGeom.Transform.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.Transform.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.Transform.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.Transform.transform = utilityCore::buildTransformationMatrix(
        newGeom.Transform.translation, newGeom.Transform.rotation, newGeom.Transform.scale);
    newGeom.Transform.inverseTransform = glm::inverse(newGeom.Transform.transform);
    newGeom.Transform.invTranspose = glm::inverseTranspose(newGeom.Transform.transform);

    objects.push_back(newGeom);
    return true;
}

int Scene::loadObject(string objectid) {
    int id = atoi(objectid.c_str());
    std::cout << "Loading Object " << id << "..." << endl;
    string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "geometry")//load geometry
        {
            loadGeometry(tokens[1], id);
        }
        else//load model
        {
            assert(tokens, size() == 3);
            assert(tokens[2] == "vnormal" || tokens[2] == "fnormal");
            bool use_vertex_normal = tokens[1] == "vnormal";
            loadModel(tokens[2], id, use_vertex_normal);
        }
    }
    return 1;
    
}



int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    std::cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        std::cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 4; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                float ior = atof(tokens[1].c_str());
                if (ior != 0.0) newMaterial.type = MaterialType::frenselSpecular;
                newMaterial.indexOfRefraction = ior;
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                float emittance = atof(tokens[1].c_str());
                if (emittance != 0.0) newMaterial.type = MaterialType::emitting;
                newMaterial.emittance = emittance;
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                float roughness = atof(tokens[1].c_str());
                if (roughness != -1.0f) newMaterial.type = MaterialType::microfacet;
                newMaterial.roughness = roughness;
            }

        }
        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::buildBVH()
{
    for (int i=0;i<objects.size();i++)
    {
        const Object& obj = objects[i];
        if (obj.type == TRIANGLE_MESH)
        {
            for (int j = obj.triangleStart; j != obj.triangleEnd; j++)
            {
                primitives.emplace_back(obj, i, j - obj.triangleStart, &triangles[0], &verticies[0]);
            }
            
        }
        else
        {
            primitives.emplace_back(obj, i);
        }
    }
    bvhroot = buildBVHTreeRecursiveSAH(primitives, 0, primitives.size(), &bvhTreeSize);
    assert(checkBVHTreeFull(bvhroot));
}

void Scene::buildStacklessBVH()
{
#if MTBVH
    compactBVHTreeToMTBVH(MTBVHArray, bvhroot, bvhTreeSize);
#else
    recursiveCompactBVHTreeForStacklessTraverse(bvhArray, bvhroot);
#endif
}
