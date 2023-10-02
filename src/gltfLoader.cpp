#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include "gltfLoader.h"

#include <iostream>

int loadGLTF(const std::string& path, std::vector<Triangle>& triangleList)
{
    // Create a TinyGLTF loader
    tinygltf::TinyGLTF loader;

    // Load the glTF file
    std::string err;
    std::string warn;
    tinygltf::Model model;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty()) {
        std::cout << "Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "Error: " << err << std::endl;
        return -1;
    }

    if (!ret) {
        std::cout << "Failed to parse glTF format" << std::endl;
        return -1;
    }

    // Get the default scene
    const tinygltf::Scene& defaultScene = model.scenes[model.defaultScene];

    // Traverse scene nodes
    for (int i = 0; i < defaultScene.nodes.size(); i++) {
        const tinygltf::Node& node = model.nodes[defaultScene.nodes[i]];

        // Check if the node is a mesh
        if (node.mesh >= 0) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];

            // Iterate through the mesh's primitives
            for (int j = 0; j < mesh.primitives.size(); j++) {
                const tinygltf::Primitive& primitive = mesh.primitives[j];

                // Check if the primitive is a triangle
                if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];

                    // Get index data
                    const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                    const unsigned char* indexBufferData = &indexBuffer.data[indexAccessor.byteOffset + indexBufferView.byteOffset];

                    // Get vertex data
                    const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
                    const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
                    const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];
                    const unsigned char* positionBufferData = &positionBuffer.data[positionAccessor.byteOffset + positionBufferView.byteOffset];

                    // Parse index data and vertex data
                    std::vector<unsigned int> indices(indexAccessor.count);
                    std::vector<float> positions(positionAccessor.count * 3);

                    for (size_t k = 0; k < indexAccessor.count; k++) {
                        if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                            const uint8_t* index = reinterpret_cast<const uint8_t*>(indexBufferData);
                            indices[k] = index[k];
                        }
                        else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            const uint16_t* index = reinterpret_cast<const uint16_t*>(indexBufferData);
                            indices[k] = index[k];
                        }
                        else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                            const uint32_t* index = reinterpret_cast<const uint32_t*>(indexBufferData);
                            indices[k] = index[k];
                        }
                    }

                    for (size_t k = 0; k < positionAccessor.count; k++) {
                        if (positionAccessor.type == TINYGLTF_TYPE_VEC3) {
                            const float* position = reinterpret_cast<const float*>(positionBufferData);
                            positions[k * 3] = position[k * 3];
                            positions[k * 3 + 1] = position[k * 3 + 1];
                            positions[k * 3 + 2] = position[k * 3 + 2];
                        }
                    }

                    // Output the vertex coordinates of the triangles
                    for (size_t k = 0; k < indices.size(); k += 3) {
                        unsigned int idx1 = indices[k];
                        unsigned int idx2 = indices[k + 1];
                        unsigned int idx3 = indices[k + 2];

                        float x1 = positions[idx1 * 3];
                        float y1 = positions[idx1 * 3 + 1];
                        float z1 = positions[idx1 * 3 + 2];

                        float x2 = positions[idx2 * 3];
                        float y2 = positions[idx2 * 3 + 1];
                        float z2 = positions[idx2 * 3 + 2];

                        float x3 = positions[idx3 * 3];
                        float y3 = positions[idx3 * 3 + 1];
                        float z3 = positions[idx3 * 3 + 2];

                        Triangle triangle;

                        triangle.v0 = { x1, y1, z1 };
                        triangle.v1 = { x2, y2, z2 };
                        triangle.v2 = { x3, y3, z3 };

                        triangleList.emplace_back(triangle);
                    }
                }
            }
        }
    }
    return 1;
}
