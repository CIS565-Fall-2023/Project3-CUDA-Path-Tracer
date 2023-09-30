#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <filesystem>
#define TINYGLTF_IMPLEMENTATION
#include "scene.h"

namespace fs = std::filesystem;
int Scene::id = 0;

std::ifstream findFile(const std::string& fileName) {
    fs::path currentPath = fs::current_path();
    for (int i = 0; i < 5; ++i) {
        fs::path filePath = currentPath / fileName;
        if (fs::exists(filePath)) {
            std::ifstream fileStream(filePath);
            if (fileStream.is_open())
                return fileStream;
        }
        currentPath = currentPath.parent_path();
    }

    std::cerr << "File not found: " << fileName << std::endl;
    return std::ifstream();
}

template<typename T>
std::pair<const T*, int> getPrimitiveBuffer(tinygltf::Model* model, const tinygltf::Primitive& primitive, const std::string& type) {
    if (primitive.attributes.find(type) == primitive.attributes.end())return{ nullptr,0 };
    const tinygltf::Accessor& accessor = model->accessors[primitive.attributes.at(type)];
    const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model->buffers[bufferView.buffer];
    const T* positions = reinterpret_cast<const T*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
    return { positions , accessor.count };
}

std::pair<const uint16_t*, int> getIndexBuffer(tinygltf::Model* model, const tinygltf::Primitive& primitive) {
    const tinygltf::Accessor& indexAccessor = model->accessors[primitive.indices];
    const tinygltf::BufferView& indexBufferView = model->bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer& indexBuffer = model->buffers[indexBufferView.buffer];
    const uint16_t* indices = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset]);
    return { indices , indexAccessor.count };
}

void updateTransform(const tinygltf::Node& node, std::vector<glm::mat4>& transforms) {
    glm::vec3 translation(0.0f);
    glm::quat rotation;
    glm::vec3 scale(1.0f);
    glm::mat4 transformation(1.0f);
    glm::mat4 t;
    if (!node.matrix.empty()) {
        transformation = glm::make_mat4(node.matrix.data());
        t = transformation;
    }

    if (!node.translation.empty()) {
        translation = glm::make_vec3(node.translation.data());
        t = glm::translate(glm::mat4(1.0f), translation);
    }

    if (!node.rotation.empty()) {
        rotation = glm::make_quat(node.rotation.data());
        t = glm::mat4_cast(rotation);
    }

    if (!node.scale.empty()) {
        scale = glm::make_vec3(node.scale.data());
        t = glm::scale(glm::mat4(1.0f), scale);
    }
    transforms.push_back(t);
}

Transformation& recomputeTransform(Transformation& t) {
    t.inverseTransform = glm::inverse(t.transform);
    t.invTranspose = glm::transpose(t.inverseTransform);
    return t;
}

Scene::Scene(std::string filename)
{
    loadSettings();
    if (settings.readFromFile)
        filename = settings.gltfPath;
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    model = new tinygltf::Model();
    bool ret = loader.LoadASCIIFromFile(model, &err, &warn, filename);
    if (!ret) {
        if (err.length() != 0)
            std::cerr << err;
        if (warn.length() != 0)
            std::cerr << warn;

        exit(-1);
    }
    createCubemapTextureObj();
    loadScene();
    tbvh = TBVH(geoms, tbb);
    if (cameras.empty()) {
        cameras.push_back(settings.defaultRenderState.camera);
    }
    auto& camera = cameras[0];
    state.camera = camera;
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    if (settings.envMapEnabled)
        loadEnvMap();
}

Scene::~Scene()
{
    for (int i = 0; i < cuda_tex_vec.size(); i++)
    {
        cudaDestroyTextureObject(cuda_tex_vec[i]);
        cudaFreeArray(dev_tex_arrs_vec[i]);
    }
    delete model;
}

Transformation evaluateTransform(std::vector<glm::mat4>& transforms) {
    Transformation t;
    t.transform = glm::mat4(1.0f);
    t.inverseTransform = glm::mat4(1.0f);
    t.invTranspose = glm::mat4(1.0f);
    for (auto it = transforms.rbegin(); it != transforms.rend(); ++it) {
        t.transform = t.transform * (*it);
    }
    t.inverseTransform = glm::inverse(t.transform);
    t.invTranspose = glm::transpose(t.inverseTransform);
    return t;
}

void Scene::traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms) {
    if (node.camera >= 0) {
        loadCamera(node, evaluateTransform(transforms).transform);
    }
    if (node.mesh >= 0) {
        meshes.emplace_back(node, evaluateTransform(transforms), this);
    }
    updateTransform(node, transforms);
    for (int childIndex : node.children) {
        if (childIndex >= 0 && childIndex < model->nodes.size()) {
            const tinygltf::Node& childNode = model->nodes[childIndex];
            traverseNode(childNode, transforms);
            transforms.pop_back();
        }
    }
}

void Scene::loadNode(const tinygltf::Node& node)
{
    std::vector<glm::mat4> transforms;
    traverseNode(node, transforms);
}


int Scene::loadScene()
{
    RenderState& state = this->state;
    state = settings.defaultRenderState;
    geoms.clear();
    int num_tex = loadTexture();
    int num_mat = loadMaterial();
    std::cout << num_mat << " materials loaded." << std::endl;
    const tinygltf::Scene& scene = model->scenes[model->defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); i++)
    {
        const tinygltf::Node& node = model->nodes[scene.nodes[i]];
        loadNode(node);
    }
    if (num_mat == 0) {
        materials.push_back(settings.defaultMat);
    }

    return 1;
}

void Scene::loadEnvMap()
{
    int width, height, numComponents;
    stbi_set_flip_vertically_on_load(true);
    float* hdrData = stbi_loadf(settings.envMapFilename.c_str(), &width, &height, &numComponents, 0);
    float* hdrDataPadded = nullptr;
    if (numComponents != 4)
    {
        std::cerr << "number of components is not 4, padded." << std::endl;
        hdrDataPadded = new float[width * height * 4];
        for (size_t i = 0; i < width * height; i++)
        {
            hdrDataPadded[i * 4] = hdrData[i * 3];
            hdrDataPadded[i * 4 + 1] = hdrData[i * 3 + 1];
            hdrDataPadded[i * 4 + 2] = hdrData[i * 3 + 2];
            hdrDataPadded[i * 4 + 3] = 1.0f;
        }
        stbi_image_free(hdrData);
    }
    else {
        hdrDataPadded = hdrData;
        numComponents = 4;
    }

    if (!hdrDataPadded) {
        std::cerr << "Error loading environment map: " << settings.envMapFilename << std::endl;
        return;
    }
    int textureIndex = textures.size();

    envMapTexture = createTextureObj(textureIndex, width, height, numComponents, hdrDataPadded, width * height * numComponents);
}

void Scene::loadSettings() {
    try {
        std::ifstream fileStream = findFile(settings.filename);
        if (!fileStream.is_open()) {
            std::cerr << "Failed to open JSON file: " << settings.filename << std::endl;
            return;
        }

        nlohmann::json jsonData;
        fileStream >> jsonData;

        RenderState& renderState = settings.defaultRenderState;
        Camera& camera = renderState.camera;
        Material& defaultMat = settings.defaultMat;
        settings.readFromFile = jsonData["GLTF"]["from file"];
        settings.gltfPath = jsonData["GLTF"]["path"];
        settings.envMapEnabled = jsonData["environmentMap"]["on"];
        settings.envMapFilename = jsonData["environmentMap"]["path"];

        nlohmann::json renderStateData = jsonData["RenderState"];
        camera.resolution.y = renderStateData["camera"]["screen height"];
        float aspectRatio = renderStateData["camera"]["aspect ratio"];
        camera.resolution.x = aspectRatio * camera.resolution.y;
        camera.position = glm::vec3(renderStateData["camera"]["position"][0],
            renderStateData["camera"]["position"][1],
            renderStateData["camera"]["position"][2]);
        camera.lookAt = glm::vec3(renderStateData["camera"]["lookAt"][0],
            renderStateData["camera"]["lookAt"][1],
            renderStateData["camera"]["lookAt"][2]);
        camera.view = glm::vec3(renderStateData["camera"]["view"][0],
            renderStateData["camera"]["view"][1],
            renderStateData["camera"]["view"][2]);
        camera.up = glm::vec3(renderStateData["camera"]["up"][0],
            renderStateData["camera"]["up"][1],
            renderStateData["camera"]["up"][2]);
        camera.fov.y = renderStateData["camera"]["fovy"];
        computeCameraParams(camera);

        settings.antiAliasing = renderStateData["antiAliasing"];
        renderState.iterations = renderStateData["iterations"];
        renderState.traceDepth = renderStateData["traceDepth"];
        renderState.imageName = renderStateData["imageName"];

        nlohmann::json materialsData = jsonData["Materials"];
        if (!materialsData.empty()) {
            defaultMat.type = materialsData[0]["type"];
            defaultMat.emissiveFactor = glm::vec3(materialsData[0]["emissiveFactor"][0],
                materialsData[0]["emissiveFactor"][1],
                materialsData[0]["emissiveFactor"][2]);
            defaultMat.alphaCutoff = materialsData[0]["alphaCutoff"];
            defaultMat.doubleSided = materialsData[0]["doubleSided"];
            defaultMat.pbrMetallicRoughness.baseColorFactor =
                glm::vec4(materialsData[0]["pbrMetallicRoughness"]["baseColorFactor"][0],
                    materialsData[0]["pbrMetallicRoughness"]["baseColorFactor"][1],
                    materialsData[0]["pbrMetallicRoughness"]["baseColorFactor"][2],
                    materialsData[0]["pbrMetallicRoughness"]["baseColorFactor"][3]);
            defaultMat.pbrMetallicRoughness.metallicFactor =
                materialsData[0]["pbrMetallicRoughness"]["metallicFactor"];
            defaultMat.pbrMetallicRoughness.roughnessFactor =
                materialsData[0]["pbrMetallicRoughness"]["roughnessFactor"];
        }
    }
    catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
}

Scene::Mesh::Mesh(const tinygltf::Node& node, const Transformation& transform, Scene* s) :scene(s)
{
    tinygltf::Model* model = s->model;
    const tinygltf::Mesh& mesh = model->meshes[node.mesh];

    glm::vec3 translation(0.0f, 0.0f, 0.0f);
    glm::quat rotation;
    glm::vec3 scale(1.0f, 1.0f, 1.0f);
    glm::mat4 modelMatrix = glm::mat4(1.0f);

    if (!node.translation.empty() || !node.rotation.empty() || !node.scale.empty()) {
        if (!node.translation.empty())
            translation = glm::make_vec3(node.translation.data());
        if (!node.rotation.empty())
            rotation = glm::make_quat(node.rotation.data());
        if (!node.scale.empty())
            scale = glm::make_vec3(node.scale.data());
        modelMatrix = glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
    }
    t.transform = transform.transform * modelMatrix;
    recomputeTransform(t);
    for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex) {
        const tinygltf::Primitive& primitive = mesh.primitives[primitiveIndex];
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
            prims.emplace_back(primitive, t, s);
        }
    }
}


Scene::Primitive::Primitive(const tinygltf::Primitive& primitive, const Transformation& t, Scene* s) :scene(s)
{
    tinygltf::Model* model = s->model;
    auto [positions, posCnt] = getPrimitiveBuffer<float>(model, primitive, "POSITION");
    auto [normals, norCnt] = getPrimitiveBuffer<float>(model, primitive, "NORMAL");
    auto [uvs, uvCnt] = getPrimitiveBuffer<float>(model, primitive, "TEXCOORD_0");
    auto [indices, indCnt] = getIndexBuffer(model, primitive);
    materialid = scene->materials.empty() ? scene->defaultMatId : primitive.material;

    for (size_t i = 0; i < indCnt; i += 3) {
        const size_t v0Id = indices[i];
        const size_t v1Id = indices[i + 1];
        const size_t v2Id = indices[i + 2];
        Triangle triangle{
            glm::vec3(positions[v0Id * 3], positions[v0Id * 3 + 1], positions[v0Id * 3 + 2]),
            glm::vec3(positions[v1Id * 3], positions[v1Id * 3 + 1], positions[v1Id * 3 + 2]),
            glm::vec3(positions[v2Id * 3], positions[v2Id * 3 + 1], positions[v2Id * 3 + 2]) };
        if (normals) {
            triangle.normal0 = glm::vec3(normals[v0Id * 3], normals[v0Id * 3 + 1], normals[v0Id * 3 + 2]);
            triangle.normal1 = glm::vec3(normals[v1Id * 3], normals[v1Id * 3 + 1], normals[v1Id * 3 + 2]);
            triangle.normal2 = glm::vec3(normals[v2Id * 3], normals[v2Id * 3 + 1], normals[v2Id * 3 + 2]);
        }
        if (uvs) {
            triangle.uv0 = glm::vec2(uvs[v0Id * 2], uvs[v0Id * 2 + 1]);
            triangle.uv1 = glm::vec2(uvs[v1Id * 2], uvs[v1Id * 2 + 1]);
            triangle.uv2 = glm::vec2(uvs[v2Id * 2], uvs[v2Id * 2 + 1]);
        }
        triangle.id = Scene::id++;
        tris.push_back(triangle);
        s->geoms.emplace_back(t, materialid, triangle.v0, triangle.v1, triangle.v2, triangle.normal0, triangle.normal1, triangle.normal2, triangle.uv0, triangle.uv1, triangle.uv2, triangle.id);
        s->tbb.expand(s->geoms.back().tbb);
    }
}

Camera& Scene::computeCameraParams(Camera& camera)const
{
    // assuming resolution, position, lookAt, view, up, fovy are already set
    float yscaled = tan(camera.fov.y * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov.x = fovx;

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);
    return camera;
}

bool Scene::loadCamera(const tinygltf::Node& node, const glm::mat4& transform)
{
    std::cout << "Loading Camera ..." << std::endl;
    Camera camera = settings.defaultRenderState.camera;
    const tinygltf::Camera& gltfCamera = model->cameras[node.camera];
    if (node.translation.size() == 3)
        camera.position = glm::vec3(transform * glm::vec4(glm::make_vec3(node.translation.data()), 1.0f));
    if (node.rotation.size() == 4)
    {
        glm::mat4 rot = glm::mat4_cast(glm::make_quat(node.rotation.data()));;
        rot = transform * rot;
        camera.view = glm::vec3(rot * glm::vec4(camera.view, 0.0f));
        camera.up = glm::vec3(rot * glm::vec4(camera.up, 0.0f));
    }
    if (gltfCamera.type == "perspective")
    {
        const tinygltf::PerspectiveCamera& perspective = gltfCamera.perspective;
        camera.fov.y = glm::degrees(perspective.yfov);
        camera.resolution.x = perspective.aspectRatio * camera.resolution.y;
    }
    else if (gltfCamera.type == "orthographic")
    {
        const tinygltf::OrthographicCamera& ortho = gltfCamera.orthographic;
        std::cout << "Orthographic Camera not implemented." << std::endl;
        return false;
    }
    cameras.push_back(computeCameraParams(camera));
    return true;
}


PbrMetallicRoughness Scene::loadPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbrMat)
{
    PbrMetallicRoughness result;
    result.baseColorFactor = glm::make_vec4(pbrMat.baseColorFactor.data());
    int textureIndex = pbrMat.baseColorTexture.index;
    if (textureIndex >= 0) {
        result.baseColorTexture = textures[textureIndex];
    }
    result.metallicFactor = pbrMat.metallicFactor;
    result.roughnessFactor = pbrMat.roughnessFactor;
    result.metallicRoughnessTexture.index = pbrMat.metallicRoughnessTexture.index;

    return result;
}

void Scene::loadExtensions(Material& material, const tinygltf::ExtensionMap& extensionMap)
{
    for (const auto& entry : extensionMap) {
        const std::string& extensionName = entry.first;
        const auto& extensionValue = entry.second;

        if (extensionName == "KHR_materials_ior") {
            material.dielectric.eta = extensionValue.Get("ior").Get<double>();
        }
        else if (extensionName == "KHR_materials_specular") {
            material.type = Material::Type::SPECULAR;
            material.dielectric.specularColorFactor = glm::vec3(extensionValue.Get("specularFactor").Get<double>());
        }
        else if (extensionName == "KHR_materials_transmission") {
            material.type = Material::Type::DIELECTRIC;
        }
        else if (extensionName == "KHR_materials_emissive_strength") {
            material.emissiveStrength = extensionValue.Get("emissiveStrength").Get<double>();
        }
        else {
            std::cerr << extensionName << " not supported." << std::endl;
        }
    }
}


int Scene::loadMaterial() {
    const std::vector<tinygltf::Material>& gltfMaterials = model->materials;
    materials.clear();

    for (size_t i = 0; i < gltfMaterials.size(); ++i) {
        const tinygltf::Material& gltfMaterial = gltfMaterials[i];
        Material material;
        material.pbrMetallicRoughness = loadPbrMetallicRoughness(gltfMaterial.pbrMetallicRoughness);

        const auto& emissiveFactor = gltfMaterial.emissiveFactor;
        material.emissiveFactor = glm::make_vec3(emissiveFactor.data());
        if (glm::length2(material.emissiveFactor) > 0.f)
            material.type = Material::Type::LIGHT;
        material.alphaCutoff = gltfMaterial.alphaCutoff;
        material.doubleSided = gltfMaterial.doubleSided;

        if (gltfMaterial.normalTexture.index >= 0)
            material.normalTexture.index = gltfMaterial.normalTexture.index;
        if (gltfMaterial.occlusionTexture.index >= 0)
            material.occlusionTexture.index = gltfMaterial.occlusionTexture.index;
        if (gltfMaterial.emissiveTexture.index >= 0)
            material.emissiveTexture.index = gltfMaterial.emissiveTexture.index;

        if (gltfMaterial.extensions.size() != 0)
            loadExtensions(material, gltfMaterial.extensions);
#ifdef DEBUG
        std::cout << "Material " << i << ": " << material.type << std::endl;
#endif
        materials.push_back(material);
    }

    return static_cast<int>(materials.size());
}

bool Scene::loadTexture() {
    int numTextures = model->textures.size();
    int totalNumTextures = numTextures + (settings.envMapEnabled ? 1 : 0);
    dev_tex_arrs_vec.resize(totalNumTextures);
    cuda_tex_vec.resize(totalNumTextures);

    for (int textureIndex = 0; textureIndex < numTextures; textureIndex++) {
        const tinygltf::Texture& texture = model->textures[textureIndex];

        if (texture.source < 0 || texture.source >= model->images.size()) {
            std::cerr << "Invalid image source for texture." << std::endl;
            return false;
        }

        const tinygltf::Image& image = model->images[texture.source];

        if (image.component != 3 && image.component != 4) {
            std::cerr << "Unsupported number of components in image (must be 3 or 4)." << std::endl;
            return false;
        }

        textures.push_back(createTextureObj(textureIndex, image.width, image.height, image.component, image.image.data(), image.image.size()));
    }
    return true;
}

template<typename T>
__host__ TextureInfo Scene::createTextureObj(int textureIndex, int width, int height, int component, const T* image, size_t size) {

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc desc;
    if (typeid(T) == typeid(unsigned char))
        desc = cudaCreateChannelDesc<uchar4>();
    else
        desc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &desc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, image, size * sizeof(T), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    if (typeid(T) == typeid(unsigned char))
        texDesc.readMode = cudaReadModeNormalizedFloat;
    else
        texDesc.readMode = cudaReadModeElementType;
    texDesc.sRGB = 1;
    texDesc.normalizedCoords = 1;
    cudaError_t cudaError;

    cudaTextureObject_t texObj = 0;
    cudaError = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error creating texture object: %s\n", cudaGetErrorString(cudaError));
    }

    dev_tex_arrs_vec[textureIndex] = cuArray;
    cuda_tex_vec[textureIndex] = texObj;

    cudaCreateTextureObject(&cuda_tex_vec[textureIndex], &resDesc, &texDesc, NULL);
    checkCUDAError("createTextureObj");
    return TextureInfo{ textureIndex, width, height, component, cuda_tex_vec[textureIndex], size };
}

void Scene::createCubemapTextureObj() {
    int cubemapData[6] = { 0, 1, 2, 3, 4, 5 };
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    cudaError_t cudaError;

    cudaError = cudaMalloc3DArray(&cubemap.cubemapArray, &channelDesc, make_cudaExtent(1, 1, 6), cudaArrayCubemap);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error cudaMalloc3DArray: %s\n", cudaGetErrorString(cudaError));
    }
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)cubemapData, sizeof(int), 1, 1);
    copyParams.dstArray = cubemap.cubemapArray;
    copyParams.extent = make_cudaExtent(1, 1, 6);
    copyParams.kind = cudaMemcpyHostToDevice;

    cudaError = cudaMemcpy3D(&copyParams);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error cudaMemcpy3D: %s\n", cudaGetErrorString(cudaError));
    }
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cubemap.cubemapArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;

    cudaError = cudaCreateTextureObject(&cubemap.texObj, &resDesc, &texDesc, nullptr);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error creating texture object: %s\n", cudaGetErrorString(cudaError));
        exit(-1);
    }
}