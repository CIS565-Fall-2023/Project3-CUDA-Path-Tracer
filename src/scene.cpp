#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "utilities.h"
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        newGeom.geomId = id;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(tokens[0].c_str(), "mesh_gltf") == 0) {
                cout << "Creating new glTF Mesh..." << endl;
                newGeom.type = MESH;
                loadMeshGltf(tokens[1].c_str(), newGeom);
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);        
#if USE_BVH
        computeAABB(newGeom);
#endif
        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
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
#if DEPTH_OF_FIELD
        else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOCALDIST") == 0) {
            camera.focalLength = atof(tokens[1].c_str());
        }
#endif
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

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}


template <typename T>
void processTris(T* indices, tinygltf::Model& model, tinygltf::Primitive& prim, tinygltf::Accessor& indexAccessor, float*& positions, Geom& gltfMesh, std::vector<Triangle>& meshTris) {
    for (size_t i = 0; i < indexAccessor.count; i += 3) {
        Triangle t;

        //vertex positions
        t.v1.pos = glm::vec3(positions[indices[i] * 3], positions[indices[i] * 3 + 1], positions[indices[i] * 3 + 2]);
        t.v2.pos = glm::vec3(positions[indices[i + 1] * 3], positions[indices[i + 1] * 3 + 1], positions[indices[i + 1] * 3 + 2]);
        t.v3.pos = glm::vec3(positions[indices[i + 2] * 3], positions[indices[i + 2] * 3 + 1], positions[indices[i + 2] * 3 + 2]);


        //albedo
        auto albedoIt = prim.attributes.find("TEXCOORD_0");
        if (albedoIt != prim.attributes.end()) {
            int albedoAccessorIndex = prim.attributes.at("TEXCOORD_0");
            tinygltf::Accessor& albedoAccessor = model.accessors[albedoAccessorIndex];
            tinygltf::BufferView& albedoBufferView = model.bufferViews[albedoAccessor.bufferView];
            tinygltf::Buffer& albedoBuffer = model.buffers[albedoBufferView.buffer];
            float* uv = reinterpret_cast<float*>(&(albedoBuffer.data[albedoBufferView.byteOffset + albedoAccessor.byteOffset]));

            t.v1.uv = glm::vec2(uv[indices[i] * 2], uv[indices[i] * 2 + 1]);
            t.v2.uv = glm::vec2(uv[indices[i + 1] * 2], uv[indices[i + 1] * 2 + 1]);
            t.v3.uv = glm::vec2(uv[indices[i + 2] * 2], uv[indices[i + 2] * 2 + 1]);

            gltfMesh.hasUVs = true;
        }


        //normals
        auto norIt = prim.attributes.find("NORMAL");
        if (norIt != prim.attributes.end()) {
            int norAccessorIndex = prim.attributes.at("NORMAL");
            tinygltf::Accessor& norAccessor = model.accessors[norAccessorIndex];
            tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
            tinygltf::Buffer& normalBuffer = model.buffers[norBufferView.buffer];
            float* normals = reinterpret_cast<float*>(&(normalBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]));

            t.v1.nor = glm::vec3(normals[indices[i] * 3], normals[indices[i] * 3 + 1], normals[indices[i] * 3 + 2]);
            t.v2.nor = glm::vec3(normals[indices[i + 1] * 3], normals[indices[i + 1] * 3 + 1], normals[indices[i + 1] * 3 + 2]);
            t.v3.nor = glm::vec3(normals[indices[i + 2] * 3], normals[indices[i + 2] * 3 + 1], normals[indices[i + 2] * 3 + 2]);

            gltfMesh.hasNormals = true;
        }
        meshTris.push_back(t);
    }
}

int Scene::loadMeshGltf(const string filename, Geom& gltfMesh) {

    tinygltf::TinyGLTF loader;
    std::string inputFilename = "../assets/" + filename;

    tinygltf::Model model;
    std::string err;
    std::string warn;
    bool success = loader.LoadASCIIFromFile(&model, &err, &warn, inputFilename.c_str());
    for (const auto& mesh : model.meshes) {               
        gltfMesh.startTriIdx = meshTris.size();
        for (auto prim : mesh.primitives) {
            int posAccessorIndex = prim.attributes.at("POSITION");
            tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
            tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            tinygltf::Buffer& positionBuffer = model.buffers[posBufferView.buffer];
            float* positions = reinterpret_cast<float*>(&(positionBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]));
            
            if (prim.material >= 0) {
                int index = model.materials[prim.material].pbrMetallicRoughness.baseColorTexture.index;
                if (index != -1) {
                    tinygltf::Texture& texture = model.textures[index];
                    std::string albedoMapPath = "../assets/" + model.images[texture.source].uri;
                    Texture tex;
                    tex.id = albedoTex.size();
                    tex.startIdx = textures.size();
                    gltfMesh.hasAlbedoMap = true;
                    gltfMesh.albedoTexId = tex.id;
                    float* albedoTexture = stbi_loadf(albedoMapPath.c_str(), &tex.width, &tex.height, &tex.numChannels, 0);
                    for (int i = 0; i < tex.width * tex.height; i++) {
                        tex.numChannels = 3;
                        glm::vec3 col = glm::vec3(albedoTexture[tex.numChannels * i], albedoTexture[tex.numChannels * i + 1], albedoTexture[tex.numChannels * i + 2]);
                        textures.push_back(col);
                    }
                    tex.endIdx = textures.size() - 1;
                    albedoTex.push_back(tex);
                    stbi_image_free(albedoTexture);
                }
            }

            if (prim.indices >= 0) {
                tinygltf::Accessor& indexAccessor = model.accessors[prim.indices];
                tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                auto type = indexAccessor.componentType;
                if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT && indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    cout << "Unsupported index type for the give glTF model." << endl;
                    return -1;
                }
                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    uint16_t* indices = reinterpret_cast<uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                    processTris(indices, model, prim, indexAccessor, positions, gltfMesh, meshTris);
                }
                else {// if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    uint32_t* indices = reinterpret_cast<uint32_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                    processTris(indices, model, prim, indexAccessor, positions, gltfMesh, meshTris);
                }                
            }            
        }
        gltfMesh.endTriIdx = meshTris.size() - 1;
        cout << "This mesh has : " << gltfMesh.endTriIdx - gltfMesh.startTriIdx + 1 << " triangles" << endl;
        cout << "Total triangles yet : " << meshTris.size() << " triangles" << endl;
    }
    return 1;
}

#if USE_BVH
void Scene::computeAABB(Geom geom) {    
    switch (geom.type) {
    case(SPHERE): {
        glm::vec3 minPos = glm::vec3(geom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.f));
        glm::vec3 maxPos = glm::vec3(geom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.f));
        boundingBoxes.push_back(AABB(minPos, maxPos, (minPos + maxPos) / 2.f, geom));
        break;
        }
    case(CUBE): {
        glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
        glm::vec3 center = glm::vec3(geom.transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
        for (float i = -0.5; i <= 0.5; i += 1.f) {
            for (float j = -0.5; j <= 0.5; j += 1.f) {
                for (float k = -0.5; k <= 0.5; k += 1.f) {
                    glm::vec3 pos = glm::vec3(geom.transform * glm::vec4(i, j, k, 1.f));
                    minPos = glm::min(minPos, pos);
                    maxPos = glm::max(maxPos, pos);
                }
            }
        }
        boundingBoxes.push_back(AABB(minPos, maxPos, center, geom));
        break;
    }
    case(MESH): {    
    for (int i = geom.startTriIdx; i <= geom.endTriIdx; i++) {
        glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
        glm::vec3 p1 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v1.pos, 1.f));
        glm::vec3 p2 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v2.pos, 1.f));
        glm::vec3 p3 = glm::vec3(geom.transform * glm::vec4(meshTris[i].v3.pos, 1.f));
        minPos = glm::min(p1, glm::min(p2, p3));
        maxPos = glm::max(p1, glm::max(p2, p3));
        glm::vec3 centroid = (p1 + p2 + p3) / 3.f;
        boundingBoxes.push_back(AABB(minPos, maxPos, centroid, geom, i));        
    }
    break;
    }
    default: cout << "Couldn't compute bounding box!" << endl;
    }
}

BVHNode::BVHNode() :boundingBox(), left(nullptr), right(nullptr) {}
BVHNode::BVHNode(AABB aabb) : boundingBox(aabb), left(nullptr), right(nullptr) {}

void BVHNode::collapseIntoSingleAABB(std::vector<AABB>& boundingBoxes) {
    glm::vec3 minPos = glm::vec3(std::numeric_limits<float>::max()), maxPos = -minPos;
    for (const AABB& aabb : boundingBoxes) {
        minPos = glm::min(minPos, aabb.minPos);
        maxPos = glm::max(maxPos, aabb.maxPos);
    }
    this->boundingBox = AABB(minPos, maxPos);
}

bool xSort(AABB a, AABB b) { return a.centroid.x < b.centroid.x; }
bool ySort(AABB a, AABB b) { return a.centroid.y < b.centroid.y; }
bool zSort(AABB a, AABB b) { return a.centroid.z < b.centroid.z; }

void splitSAH(std::vector<AABB>& boundingBoxes) {
    std::sort(boundingBoxes.begin(), boundingBoxes.end(), xSort);
    float xLen = boundingBoxes.back().centroid.x - boundingBoxes.front().centroid.x;
    std::sort(boundingBoxes.begin(), boundingBoxes.end(), ySort);
    float yLen = boundingBoxes.back().centroid.y - boundingBoxes.front().centroid.y;
    std::sort(boundingBoxes.begin(), boundingBoxes.end(), zSort);
    float zLen = boundingBoxes.back().centroid.z - boundingBoxes.front().centroid.z;
    if (xLen > yLen && xLen > zLen) {
        std::sort(boundingBoxes.begin(), boundingBoxes.end(), xSort);
    }
    else if (yLen >= xLen && yLen >= zLen) {
        std::sort(boundingBoxes.begin(), boundingBoxes.end(), ySort);
    }
}

void buildBVH(BVHNode*& node, std::vector<AABB>& boundingBoxes) {
    node = new BVHNode();    
    node->collapseIntoSingleAABB(boundingBoxes);
    if (boundingBoxes.size() <= 2) {
        node->left = new BVHNode(boundingBoxes[0]);
        if (boundingBoxes.size() == 2) {
            node->right = new BVHNode(boundingBoxes[1]);            
        }
        return;
    }
    else {
        splitSAH(boundingBoxes);
        node->collapseIntoSingleAABB(boundingBoxes);
        int splitIdx = floor(boundingBoxes.size() / 2.f);
        buildBVH(node->left, std::vector<AABB>(boundingBoxes.begin(), boundingBoxes.begin() + splitIdx));
        buildBVH(node->right, std::vector<AABB>(boundingBoxes.begin() + splitIdx, boundingBoxes.end()));
    }
}

void nofOfNodesInBVH(BVHNode* node, int& count) {
    count++;
    if (node->left != __nullptr) {
        nofOfNodesInBVH(node->left, count);
    }
    if (node->right != __nullptr) {
        nofOfNodesInBVH(node->right, count);
    }
}

int flattenBVH(std::vector<LBVHNode>& flattenedBVH, BVHNode* node, int& offset) {
    int currentOffset = offset++;
    flattenedBVH[currentOffset].boundingBox = node->boundingBox;
    if (node->left) {
        flattenBVH(flattenedBVH, node->left, offset);
        if (node->right) {
            flattenedBVH[currentOffset].secondChildOffset = flattenBVH(flattenedBVH, node->right, offset);
        }
    }
    else {
        flattenedBVH[currentOffset].isLeaf = true;
    }
    return currentOffset;
}
#endif
