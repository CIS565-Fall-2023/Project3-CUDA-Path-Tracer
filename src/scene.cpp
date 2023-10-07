#define TINYGLTF_IMPLEMENTATION
#define INTEGER_ADDRESS 0

#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_gltf.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    hasEnvMap = false;
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
            else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
                loadMesh(tokens[1]);
                cout << "mesh loading in progress" << endl;
            }
            else if (strcmp(tokens[0].c_str(), "EMAP") == 0) {
                loadEnv(tokens[1]);
                cout << "environment map loading in progress" << endl;
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

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
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

int Scene::loadMesh(string meshid) {

    int id = atoi(meshid.c_str());
    string address;
    string line;
    utilityCore::safeGetline(fp_in, address);
    address = "../scenes/" + address;
    char* fname = (char*)address.c_str();
    if (fp_in.good()) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        string err;
        string warn;

        bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, address);

        Geom newGeom;
        newGeom.type = MESH;
        newGeom.triIdx = tris.size();
        newGeom.triCount = 0;
        int offset = 0;
        for (const auto& modelMesh : model.meshes) {
            for (const auto& p : modelMesh.primitives) {

                // READING INDEX BUFFER FOR TRIANGLE INDICES

                const auto& indicesAccessor = model.accessors[p.indices];
                const auto& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
                const auto& buffer = model.buffers[indicesBufferView.buffer];

                auto rawData = buffer.data;
                auto start = indicesBufferView.byteOffset + indicesAccessor.byteOffset;
                const auto idxCount = indicesAccessor.count;

#if INTEGER_ADDRESS
                
                if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) throw;
                cout << "using int" << endl;
                unsigned int* indicesBuffer = reinterpret_cast<unsigned int*>(&rawData[start]);

#else
                cout << "using short" << endl;
                if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) throw;
                unsigned short* indicesBuffer = reinterpret_cast<unsigned short*>(&rawData[start]);
#endif

                // READING POSITION, NORMAL, UV BUFFER
                float* posBuffer;
                float* norBuffer;
                float* uvBuffer;

                for (const auto& att: p.attributes) {
                    const auto& attAccessor = model.accessors[att.second];
                    const auto& attBufferView = model.bufferViews[attAccessor.bufferView];
                    const auto& attBuffer = model.buffers[attBufferView.buffer];

                    rawData = attBuffer.data;
                    start  = attBufferView.byteOffset + attAccessor.byteOffset;
                    if (att.first == "POSITION") {
                        posBuffer = reinterpret_cast<float*>(&rawData[start]);
                    }
                    else if (att.first == "NORMAL") {
                        norBuffer = reinterpret_cast<float*>(&rawData[start]);
                    }
                    else {
                        uvBuffer = reinterpret_cast<float*>(&rawData[start]);
                    }
                }
                                
                // Load pos, nor, and uv data into triangle struct
                for (size_t i = 0; i < idxCount; i += 3) {
                    Triangle t;
                    for (int j = 0; j < 3; j++) {
                        Vertex v;
                        auto currIdx = indicesBuffer[i + j];
                        v.pos = glm::vec3(posBuffer[3 * currIdx], posBuffer[3 * currIdx + 1], posBuffer[3 * currIdx + 2]);
                        v.nor = glm::vec3(norBuffer[3 * currIdx], norBuffer[3 * currIdx + 1], norBuffer[3 * currIdx + 2]);
                        v.uv = glm::vec2(uvBuffer[2 * currIdx], uvBuffer[2 * currIdx + 1]);
                        t.vertices[j] = v;
                    }
                    tris.push_back(t);
                    newGeom.triCount++;
                }
                offset += 1;
            }
        }
        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        geoms.push_back(newGeom);
    }
    return 1;
}
int Scene::loadEnv(string envid) {

    int id = atoi(envid.c_str());
    string address;
    string line;
    utilityCore::safeGetline(fp_in, address);
    address = "../scenes/" + address;
    char* fname = (char*)address.c_str();
    if (fp_in.good()) {
        float* img = stbi_loadf(fname, &mp.width, &mp.height, &mp.channels, 0);
        for (size_t i = 0; i < mp.width * mp.height; i++) {
            for (size_t c = 0; c < mp.channels; c++) {
                mp.imgdata.push_back(img[mp.channels * i + c]);
            }
            for (size_t c = mp.channels; c < 4; c++) {
                mp.imgdata.push_back(1.0f);
            }
        }
        hasEnvMap = true;
    }
    return 1;
}
