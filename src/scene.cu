#include "scene.h"
#include "sceneStructs.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

void Scene::traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4x4 & parentTransform)
{
    if (nodeIndex < 0 || nodeIndex >= model.nodes.size()) {
        return;
    }

    const auto& node = model.nodes[nodeIndex];
    
    glm::mat4 nodeTransform = parentTransform;
    applyNodeTransform(node, nodeTransform);

    if (node.mesh >= 0) {
        const auto& mesh = model.meshes[node.mesh];
        processMesh(model, mesh, nodeTransform);
    }

    // Recursively process child nodes
    for (int childIndex : node.children) {
        traverseNode(model, childIndex, nodeTransform);
    }
}

void Scene::processMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4x4 & transform)
{
    std::cout << "Loading mesh: " << mesh.name << std::endl;

    for (const auto& primitive : mesh.primitives) {
        int p_size = mesh.primitives.size();
        const auto& indicesAccessor = model.accessors[primitive.indices];
        const auto& positionsAccessor = model.accessors[primitive.attributes.at("POSITION")];
        const auto& normalsAccessor = model.accessors[primitive.attributes.at("NORMAL")];
        const auto& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];

        const auto& indicesView = model.bufferViews[indicesAccessor.bufferView];
        const auto& positionsView = model.bufferViews[positionsAccessor.bufferView];
        const auto& normalsView = model.bufferViews[normalsAccessor.bufferView];
        const auto& uvView = model.bufferViews[uvAccessor.bufferView];

        // TODO: Dynamic type array according to componentType
        const unsigned short* indexData = reinterpret_cast<const unsigned short*>(&model.buffers[indicesView.buffer].data[indicesAccessor.byteOffset + indicesView.byteOffset]);
        const float* positionData = reinterpret_cast<const float*>(&model.buffers[positionsView.buffer].data[positionsAccessor.byteOffset + positionsView.byteOffset]);
        const float* normalData = reinterpret_cast<const float*>(&model.buffers[normalsView.buffer].data[normalsAccessor.byteOffset + normalsView.byteOffset]);
        const float* uvData = reinterpret_cast<const float*>(&model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]);

        const size_t vertexStride = 3;
        const size_t normalStride = 3;
        const size_t uvStride = 3;

        const size_t numIndices = indicesAccessor.count;
        glm::mat4x4 normalTransform = glm::transpose(glm::inverse(transform));
        // Iterate through indices and create triangles
        //for (size_t i = 0; i < numIndices; i += 3) {
        for (size_t i = 0; i < numIndices; i += 3) {
            Triangle triangle;
            triangle.p1 = glm::vec3(positionData[indexData[i] * vertexStride], positionData[indexData[i] * vertexStride + 1], positionData[indexData[i] * vertexStride + 2]);
            triangle.p2 = glm::vec3(positionData[indexData[i + 1] * vertexStride], positionData[indexData[i + 1] * vertexStride + 1], positionData[indexData[i + 1] * vertexStride + 2]);
            triangle.p3 = glm::vec3(positionData[indexData[i + 2] * vertexStride], positionData[indexData[i + 2] * vertexStride + 1], positionData[indexData[i + 2] * vertexStride + 2]);
            
            triangle.p1 = glm::vec3(transform * glm::vec4(triangle.p1, 1.0f));
            triangle.p2 = glm::vec3(transform * glm::vec4(triangle.p2, 1.0f));
            triangle.p3 = glm::vec3(transform * glm::vec4(triangle.p3, 1.0f));

            triangle.n1 = glm::vec3(normalData[indexData[i] * normalStride], normalData[indexData[i] * normalStride + 1], normalData[indexData[i] * normalStride + 2]);
            triangle.n2 = glm::vec3(normalData[indexData[i + 1] * normalStride], normalData[indexData[i + 1] * normalStride + 1], normalData[indexData[i + 1] * normalStride + 2]);
            triangle.n3 = glm::vec3(normalData[indexData[i + 2] * normalStride], normalData[indexData[i + 2] * normalStride + 1], normalData[indexData[i + 2] * normalStride + 2]);

            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));
            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));
            triangle.n1 = glm::vec3(normalTransform * glm::vec4(triangle.n1, 0.0f));

            triangle.uv1 = glm::vec2(uvData[indexData[i] * uvStride], uvData[indexData[i] * uvStride + 1]);
            triangle.uv2 = glm::vec2(uvData[indexData[i + 1] * uvStride], uvData[indexData[i + 1] * uvStride + 1]);
            triangle.uv3 = glm::vec2(uvData[indexData[i + 2] * uvStride], uvData[indexData[i + 2] * uvStride + 1]);
            //auto index0 = indexData[i];
            //auto index1 = indexData[i + 1];
            //auto index2 = indexData[i + 2];
            triangles.push_back(triangle);
        }
    }
}

Scene::Scene(const char* filename)
{
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    //bool success = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    bool success = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        std::cout << "Warn: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
        assert(0);
    }

    if (!success) {
        std::cerr << "Failed to load glTF model." << std::endl;
        assert(0);
    }

    initTriangles();
}

void Scene::applyNodeTransform(const tinygltf::Node & node, glm::mat4x4& parentTransform)
{
    glm::mat4 localTransform(1.0f);
    glm::mat4 T(1.0f), R(1.0f), S(1.0f);

    if (!node.translation.empty()) {
        const auto& translation = node.translation;
        T = glm::translate(localTransform, glm::vec3(translation[0], translation[1], translation[2]));
    }

    if (!node.rotation.empty()) {
        const auto& rotation = node.rotation;
        // Please note: 
        // tinygltf stores quaternions in the order w, x, y, z
        // glm::mat constructor in the order x, y, z, w (It seems that glm quat module is not even internally consistent...)
        // Ref: https://stackoverflow.com/questions/48348509/glmquat-why-the-order-of-x-y-z-w-components-are-mixed
        glm::quat rotationQuaternion(rotation[3], rotation[0], rotation[1], rotation[2]);
        R = glm::mat4_cast(rotationQuaternion);
    }

    if (!node.scale.empty()) {
        const auto& scale = node.scale;
        S = glm::scale(localTransform, glm::vec3(scale[0], scale[1], scale[2]));
    }
    localTransform = T * R * S;
    // Update the parent transformation matrix with the node's transformation
    parentTransform = parentTransform * localTransform;
}

void Scene::initTriangles()
{
    auto initTransform = glm::mat4x4(1.0f);
    for (auto node : model.scenes[0].nodes) {
        traverseNode(model, node, initTransform);
    }
    return;
}

__global__ void freePrimitive(Primitive** primitives, int size) {
    for (size_t i = 0; i < size; i++)
    {
        delete primitives[i];
    }
}

void Scene::freeBuffer()
{
    int blockSize = 128;
    dim3 loadPrimitiveBlock((triangles.size() + (blockSize - 1)) / blockSize);
    //freePrimitive << <loadPrimitiveBlock, blockSize >> > (dev_primitives, getPrimitiveSize());
    freePrimitive << <1, 1>> > (dev_primitives, getPrimitiveSize());
    //for (size_t i = 0; i < getPrimitiveSize(); i++)
    //{
    //    delete dev_primitives[i];
    //}
    cudaFree(dev_primitives);
}
