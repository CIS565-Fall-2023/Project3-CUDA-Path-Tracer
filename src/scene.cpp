#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYGLTF_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"


static string GetFilePathExtension(const string &fileName) {
	if (fileName.find_last_of(".") != string::npos)
		return fileName.substr(fileName.find_last_of(".") + 1);
	return "";
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;

    std::string ext = GetFilePathExtension(filename);
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
	if (ext.compare("gltf") == 0) {
        cout << "Reading gltf scene file" << endl;
		ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        ret = parseGLTFModel(model);
	}
	else if (ext.compare("glb") == 0) {
        cout << "Reading glb scene file" << endl;
		ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
        ret = parseGLTFModel(model);
	}
    else {
        cout << "Reading generic scene file" << endl;
        ret = parseGenericFile(filename);
    }

    if (!warn.empty()) {
        cout << "Warn: " << warn << endl;
    }
    else if (!err.empty()) {
        cout << "Err: " << err << endl;
    }
    else if (!ret) {
        cout << "Failed to parse gltf file" << endl;
    }
    else {
        cout << "Successfully parsed gltf file" << endl;
    }
}

int Scene::parseGLTFModel(const tinygltf::Model &model) {
    auto& scenes = model.scenes;

    for (auto& scene : scenes) {
        auto& nodes = scene.nodes;
        for (int node : nodes) {
            auto& nodeObj = model.nodes[node];

            glm::vec3 translation(0.0f);
            if (nodeObj.translation.size() == 3) {
                translation = glm::vec3(nodeObj.translation[0], nodeObj.translation[1], nodeObj.translation[2]);
            }

            glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
            if (nodeObj.rotation.size() == 4) {
                rotation = glm::quat(nodeObj.rotation[3], nodeObj.rotation[0], nodeObj.rotation[1], nodeObj.rotation[2]);
            }

            glm::vec3 scale(1.0f);
            if (nodeObj.scale.size() == 3) {
                scale = glm::vec3(nodeObj.scale[0], nodeObj.scale[1], nodeObj.scale[2]);
            }

            glm::mat4 transform = glm::translate(glm::mat4(), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
            glm::mat4 inverseTransform = glm::inverse(transform);
            glm::mat4 invTranspose = glm::inverseTranspose(transform);
            
            // print transform
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4;j++) {
                    std::cout << transform[i][j] << " ";
                }
                std::cout << std::endl;
            }
            // if nodeObj has mesh
            if (nodeObj.mesh != -1) {
                int mesh = nodeObj.mesh;
                std::cout << "node: " << node << std::endl;
                auto& meshObj = model.meshes[mesh];
                std::cout << "mesh: " << mesh << std::endl;
                auto& primitives = meshObj.primitives;
                for (auto& primitive : primitives) {
                    auto& attributes = primitive.attributes;
                    auto it = attributes.find("POSITION");
                    if (it == attributes.end()) {
                        std::cout << "no position attribute" << std::endl;
                        continue;
                    }
                    auto& posAccessor = model.accessors[it->second];
                    auto& posBufferView = model.bufferViews[posAccessor.bufferView];
                    auto& posBuffer = model.buffers[posBufferView.buffer].data;

                    std::vector<float> vertices(posAccessor.count * 3);
                    std::memcpy(vertices.data(), posBuffer.data() + posBufferView.byteOffset + posAccessor.byteOffset, posAccessor.count * 3 * sizeof(float));

                    auto& indices = primitive.indices;      
                    auto& indexAccessor = model.accessors[indices];
                    auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                    auto& indexBuffer = model.buffers[indexBufferView.buffer].data;

                    std::vector<unsigned short> indicesVec(indexAccessor.count);
                    std::memcpy(indicesVec.data(), indexBuffer.data() + indexBufferView.byteOffset + indexAccessor.byteOffset, indexAccessor.count * sizeof(unsigned short));

                    Mesh newMesh;
                    newMesh.vertices = vertices;
                    newMesh.indices = indicesVec;
                    newMesh.transform = transform;
                    newMesh.inverseTransform = inverseTransform;
                    newMesh.invTranspose = invTranspose;
                    newMesh.translation = translation;
                    newMesh.rotation = rotation;
                    newMesh.scale = scale;
                    meshes.push_back(newMesh);
                    std::cout << "vertices: " << std::endl;
                    for (int i = 0; i < vertices.size(); i += 3) {
                        std::cout << vertices[i] << " " << vertices[i + 1] << " " << vertices[i + 2] << std::endl;
                    }
                    std::cout << "indices: " << std::endl;
                    for (int i = 0; i < indicesVec.size(); i += 3) {
                        std::cout << indicesVec[i] << " " << indicesVec[i + 1] << " " << indicesVec[i + 2] << std::endl;
                    }
                }
            }
            else if (nodeObj.camera != -1) {
                int camera = nodeObj.camera;
                auto& cameraObj = model.cameras[camera];
                std::cout << "camera: " << camera << std::endl;
                if (cameraObj.type == "perspective") {
                    loadGLTFPerspectiveCamera(cameraObj, translation);
                }
            }
        }
    }
    return 1;
}

int Scene::loadGLTFPerspectiveCamera(const tinygltf::Camera &cameraObj, glm::vec3& translation) {
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy = cameraObj.perspective.yfov * (180 / PI);

    camera.resolution.x = 800; // TODO: derive from cameraObj
    camera.resolution.y = 800;
    float aspectRatio = cameraObj.perspective.aspectRatio;
    camera.fov = glm::vec2(fovy * aspectRatio, fovy);
    camera.position = translation;
    camera.lookAt = glm::vec3(0.0f, 5.0f, 0.0f); // TODO: derive from cameraObj
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f); // TODO: derive from cameraObj
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded gltf camera!" << endl;
    return 1;
}

int Scene::parseGenericFile(string filename) {
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
    return 1;
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
