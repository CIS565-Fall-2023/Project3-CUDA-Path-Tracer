#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <memory>
#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "tinygltf/tiny_gltf.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

Scene::Scene(string filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                loadTexture(tokens[1]);
                std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                std::cout << " " << std::endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                std::cout << " " << std::endl;
            }
        }
    }
}

bool Scene::loadOBJ(const std::string& filePath, GLTFMesh& gltfMesh)
{
    tinyobj::ObjReaderConfig reader_config;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filePath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    float maxX = -FLT_MAX;
    float maxY = -FLT_MAX;
    float maxZ = -FLT_MAX;
    float minX = FLT_MAX;
    float minY = FLT_MAX;
    float minZ = FLT_MAX;
    std::vector<glm::vec3> tempVertices;
    std::vector<glm::vec3> tempNormals;
    std::vector<glm::vec2> tempUVs;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                glm::vec3 vertex;
                glm::vec3 normal;
                glm::vec2 uv;
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                vertex = glm::vec3(vx, vy, vz);
                maxX = glm::max(vx, maxX);
                maxY = glm::max(vy, maxY);
                maxZ = glm::max(vz, maxZ);
                minX = glm::min(vx, minX);
                minY = glm::min(vy, minY);
                minZ = glm::min(vz, minZ);
                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normal = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    // for obj its y should be 1-y
                    uv = glm::vec2(tx, 1.0f - ty);
                }
                tempVertices.push_back(vertex);
                tempNormals.push_back(normal);
                tempUVs.push_back(uv);
            }
            index_offset += fv;



            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    for (int i = 0; i < tempVertices.size(); i = i + 3)
    {
        gltfMesh.triangles.push_back(Triangle{
                        tempVertices[i],
                        tempVertices[i + 1],
                        tempVertices[i + 2],
                        tempNormals[i],
                        tempNormals[i + 1],
                        tempNormals[i + 2],
                        tempUVs[i],
                        tempUVs[i + 1],
                        tempUVs[i + 2],
            });
    }

    gltfMesh.bbmin = glm::vec3(minX, minY, minZ);
    gltfMesh.bbmax = glm::vec3(maxX, maxY, maxZ);
}


//Assume the scales for the mesh are all uniformed, so that the normal don't need to be changed
bool Scene::loadGLTF(const std::string& filePath, GLTFMesh& gltfMesh, glm::vec3 trans, glm::vec3 scale)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    //for glft, not glb
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filePath.c_str());

    if (!warn.empty()) {
        std::cout << "glTF parse warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "glTF parse error: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load glTF: " << filePath << std::endl;
        return false;
    }
    glm::vec3 tempMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 tempMax = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    //GLFTMesh loadMesh;
    for (int i = 0; i < model.meshes.size(); i++)
    {
        const auto& mesh = model.meshes[i];
        // for each primitives
        for (int j = 0; j < mesh.primitives.size(); j++)
        {
            const auto& primitive = mesh.primitives[j];
            std::vector<int> faceIndex;
            std::vector<glm::vec3> tempVertices;
            std::vector<glm::vec3> tempNormals;
            std::vector<glm::vec2> tempUVs;
            //face index
            if (primitive.indices >= 0)
            {
                const auto& indexAccessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[indexAccessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const auto indexStart = buffer.data.data() + bufferView.byteOffset +
                    indexAccessor.byteOffset;
                const auto byteStride = indexAccessor.ByteStride(bufferView);
                const auto count = indexAccessor.count;

                //const unsigned char* indexPtr = indexStart;
                switch (indexAccessor.componentType)
                {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: 
                {
                    const unsigned char* indexPtr = indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                {
                    const unsigned short* indexPtr = (unsigned short*) indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                {
                    const unsigned int* indexPtr = (unsigned int*)indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
                case TINYGLTF_COMPONENT_TYPE_BYTE:
                {
                    const char* indexPtr = (char*)indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
                case TINYGLTF_COMPONENT_TYPE_SHORT:
                {
                    const short* indexPtr = (short*)indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
                case TINYGLTF_COMPONENT_TYPE_INT:
                {
                    const int* indexPtr = (int*)indexStart;
                    for (int ind = 0; ind < count; ind++)
                    {
                        faceIndex.push_back((int)indexPtr[ind]);
                    }
                }; break;
            }
            }
            
            // only triangle mode is supported
            if (primitive.mode == TINYGLTF_MODE_TRIANGLES)
            {
                //find the right attributes to obtain the mesh information


                //vertex position
                const auto& posInfo = primitive.attributes.find("POSITION");
                const auto& posIndex = (*posInfo).second;

                const auto& posAccessor = model.accessors[posIndex];
                const auto& posBufferView = model.bufferViews[posAccessor.bufferView];
                const auto& posBuffer = model.buffers[posBufferView.buffer];
                const auto& posStart = posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset;
                int byteStride = posAccessor.ByteStride(posBufferView);
                int count = posAccessor.count;

                gltfMesh.faceCount = count;

                const unsigned char* posPtr = posStart;
                for (int ind = 0; ind < count; ind++)
                {
                    glm::vec3 t = *(reinterpret_cast<const glm::vec3*>(posStart + ind * byteStride));
                    tempVertices.push_back(glm::vec3(t.x, t.y, t.z));
                    tempMax.x = glm::max(t.x, tempMax.x);
                    tempMax.y = glm::max(t.y, tempMax.y);
                    tempMax.z = glm::max(t.z, tempMax.z);
                    tempMin.x = glm::min(t.x, tempMin.x);
                    tempMin.y = glm::min(t.y, tempMin.y);
                    tempMin.z = glm::min(t.z, tempMin.z);
                }
                //normal position
                const auto& normalInfo = primitive.attributes.find("NORMAL");
                const auto& normalIndex = (*normalInfo).second;
                const auto& normalAccessor = model.accessors[normalIndex];
                const auto& normalBufferView = model.bufferViews[normalAccessor.bufferView];
                const auto& normalBuffer = model.buffers[normalBufferView.buffer];
                const auto& normalStart = normalBuffer.data.data() + normalBufferView.byteOffset + normalAccessor.byteOffset;
                byteStride = normalAccessor.ByteStride(normalBufferView);
                count = normalAccessor.count;
                const unsigned char* ptr = normalStart;
                for (int ind = 0; ind < count; ind++)
                {
                    glm::vec3 t = *(reinterpret_cast<const glm::vec3*>(normalStart + ind * byteStride));
                    tempNormals.push_back(glm::vec3(t.x, t.y, t.z));
                }

                //texture position
                const auto& textureInfo = primitive.attributes.find("TEXCOORD_0");
                const auto& textureIndex = (*textureInfo).second;
                const auto& textureAccessor = model.accessors[textureIndex];
                const auto& textureBufferView = model.bufferViews[textureAccessor.bufferView];
                const auto& textureBuffer = model.buffers[textureBufferView.buffer];
                const auto& textureStart = textureBuffer.data.data() + textureBufferView.byteOffset + textureAccessor.byteOffset;
                byteStride = textureAccessor.ByteStride(textureBufferView);
                count = textureAccessor.count;
                const unsigned char* texturePtr = textureStart;
                for (int ind = 0; ind < count; ++ind)
                {
                    glm::vec2 t = *(reinterpret_cast<const glm::vec2*>(textureStart + ind * byteStride));
                    //load uv
                    tempUVs.push_back(glm::vec2(t[0], t[1]));
                }


                for (int i = 0; i < faceIndex.size(); i = i + 3)
                {
                    gltfMesh.triangles.push_back(Triangle{
                        tempVertices[faceIndex[i]],
                        tempVertices[faceIndex[i + 1]],
                        tempVertices[faceIndex[i + 2]],
                        tempNormals[faceIndex[i]],
                        tempNormals[faceIndex[i + 1]],
                        tempNormals[faceIndex[i + 2]],
                        tempUVs[faceIndex[i]],
                        tempUVs[faceIndex[i + 1]],
                        tempUVs[faceIndex[i + 2]],
                        });
                }

            }
            else
            {
                std::cout << "Not triangle mode, not supported" << std::endl;
            }
        }
    }
    std::cout << gltfMesh.triangles.size() << std::endl;
    gltfMesh.bbmax = tempMax;
    gltfMesh.bbmin = tempMin;
    std::cout << gltfMesh.bbmax.x << "," << gltfMesh.bbmax.y << "," << gltfMesh.bbmax.z << std::endl;
    std::cout << gltfMesh.bbmin.x << "," << gltfMesh.bbmin.y << "," << gltfMesh.bbmin.z << std::endl;
    return true;
}


int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Geom " << id << "..." << std::endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                std::cout << "Creating new sphere..." << std::endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                std::cout << "Creating new cube..." << std::endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) {
                std::cout << "Creating new mesh..." << std::endl;
                newGeom.type = MESH;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << std::endl;
        }

        //link texture
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            // if have no texture then texture id would be -1
            newGeom.textureid = atoi(tokens[1].c_str());
            std::cout << "Connecting Geom " << objectid << " to Texture " << newGeom.textureid << "..." << std::endl;
        }

        int meshID = 0;
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
            else if (strcmp(tokens[0].c_str(), "LINK") == 0) {
                newGeom.meshid = meshID;
                //TODO: read mesh
                std::string path = tokens[1].c_str();
                std::vector<int> offs;
                std::vector<Triangle> triangles;
                glm::vec3 min;
                glm::vec3 max;
                GLTFMesh mesh1 = {
                    min,
                    max,
                    0,
                    offs,
                    triangles
                };
                std::string ext;
                if (path.find_last_of(".") != std::string::npos)
                    ext =  path.substr(path.find_last_of(".") + 1);
                if (ext.compare("gltf") == 0) {
                    // assume glTF.
                    loadGLTF(path, mesh1, newGeom.translation, newGeom.scale);
                }
                else if(ext.compare("obj") == 0) {
                    // assume glTF.
                    loadOBJ(path, mesh1);
                }
                gltfMeshes.push_back(mesh1);
                ++meshID;
            }
            
            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << std::endl;
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

    std::cout << "Loaded camera!" << std::endl;
    return 1;
}

int Scene::loadTexture(string textureID) {
    int id = atoi(textureID.c_str());
    int tmpStart = texData.size();
    int tmpEnd = texData.size();

    int width = 0;
    int height = 0;
    int channel = 0;
    if (id != textures.size()) {
        std::cout << "ERROR: TEXTURE ID does not match expected number of textures" << std::endl;
        return -1;
    }
    else {
        std::cout << "Loading Texture " << id << "..." << std::endl;
        Texture newTexture;
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        const char* texturePath;
        texturePath = tokens[1].c_str();
        unsigned char* pixels = stbi_load(texturePath, &width, &height, &channel, 0);
        // if there isn't a valid path then return a procedural texture
        if (height == 0 || width == 0)
        {
            cout << "ssssss" << endl;
            width = 512;
            height = 512;
            std::vector<glm::vec2> randCenter;
            for (int ind = 0; ind < 100; ind++)
            {
                float x = (float)rand() / RAND_MAX;
                float y = (float)rand() / RAND_MAX;
                randCenter.push_back(glm::vec2(x, y));
            }
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float u = j / 512.0f;
                    float v = i / 512.0f;
                    float minDistance = FLT_MAX;
                    for (glm::vec2 rcenter : randCenter)
                    {
                        float distance = glm::length(glm::vec2(u, v) - rcenter);
                        minDistance = std::min(distance, minDistance);
                    }
                    texData.push_back(glm::vec3(minDistance * 7.6f, minDistance * 5.6f, minDistance * 3.6f));
                }
            }
        }
        else
        {
            for (int i = 0; i < width * height - 1; i++)
            {
                // only accept 3 channels rgb here
                if (channel == 3)
                {
                    texData.push_back(glm::vec3((int(pixels[i * 3]) / 255.0f), (int(pixels[i * 3 + 1]) / 255.0f), (int(pixels[i * 3 + 2]) / 255.0f)));
                }
                else if (channel == 4)
                {
                    texData.push_back(glm::vec3((int(pixels[i * 4]) / 255.0f), (int(pixels[i * 4 + 1]) / 255.0f), (int(pixels[i * 4 + 2]) / 255.0f)));
                }
                else
                {
                    std::cout << "Please input valid image texture!";
                }
            }
        }
        stbi_image_free(pixels);
        
        //if start == end, then the texture is invalid
        newTexture.height = height;
        newTexture.width = width;
        newTexture.texID = id;
        newTexture.start = tmpStart;
        newTexture.end = texData.size();
        
        textures.push_back(newTexture);
        return 1;
    }
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Material " << id << "..." << std::endl;
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
