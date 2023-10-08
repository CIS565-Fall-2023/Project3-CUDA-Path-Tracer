#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tinyobj/tiny_obj_loader.h"

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
            } else if (strcmp(tokens[0].c_str(), "MESH") == 0) {
                loadMesh(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                loadTexture(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "BACKGROUND") == 0) {
                loadEnvMap();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadMesh(string objectid)
{
    int id = atoi(objectid.c_str());
    if (id != meshs.size()) {
        cout << "ERROR: MESH ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Mesh " << id << "..." << endl;
        Mesh newMesh;
        string line;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        string filename = ROOT_PATH;
        filename += "/objs/";
        if (!line.empty() && fp_in.good()) {
            filename += line;
            cout << "Creating new mesh from " << filename <<endl;
            string res = tinyobj::LoadObj(shapes, materials, filename.c_str());
            if (res.size() == 0 && shapes.size() > 0) {
                vector<float>& pos = shapes[0].mesh.positions;
                vector<float>& norms = shapes[0].mesh.normals;
                vector<float>& uvs = shapes[0].mesh.texcoords;
                vector<unsigned int>& idxs = shapes[0].mesh.indices;

                int n;
                vector<glm::vec3> v_pos;
                n = pos.size();
                for (int i = 0;i < n;i += 3) {
                    v_pos.push_back(glm::vec3(pos[i],pos[i+1],pos[i+2]));
                }
                
                vector<glm::vec3> v_norm;
                n = norms.size();
                for (int i = 0;i < n;i += 3) {
                    v_norm.push_back(glm::vec3(norms[i], norms[i + 1], norms[i + 2]));
                }

                vector<glm::vec2> v_uv;
                n = uvs.size();
                for (int i = 0;i < n;i += 2) {
                    v_uv.push_back(glm::vec2(uvs[i], uvs[i + 1]));
                }
                n = idxs.size();
                for (int i = 0;i < n;i += 3) {
                    
                    int idx1 = idxs[i];
                    int idx2 = idxs[i + 1];
                    int idx3 = idxs[i + 2];

                    Vertex v1{ v_pos[idx1],v_norm[idx1],v_uv[idx1] };
                    Vertex v2{ v_pos[idx2],v_norm[idx2],v_uv[idx2] };
                    Vertex v3{ v_pos[idx3],v_norm[idx3],v_uv[idx3] };

                    trigs.push_back({ v1,v2,v3,id });
                }
            }
            else {
                cout << "Error with tinyobj: " << res << endl;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newMesh.materialId = atoi(tokens[1].c_str());
            cout << "Connecting Mesh " << objectid << " to Material " << newMesh.materialId << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newMesh.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newMesh.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newMesh.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newMesh.transform = utilityCore::buildTransformationMatrix(
            newMesh.translation, newMesh.rotation, newMesh.scale);
        newMesh.inverseTransform = glm::inverse(newMesh.transform);
        newMesh.invTranspose = glm::inverseTranspose(newMesh.transform);

        meshs.push_back(newMesh);
        return 1;
    }
    return 0;
}

int Scene::loadTexture(string textureid)
{
    int id = atoi(textureid.c_str());
    if (id != texs.size()) {
        cout << "ERROR: TEXTURE ID does not match expected number of geoms" << endl;
        return -1;
    }
    else
    {
        cout << "Loading Texture " << id << "..." << endl;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        string filename = ROOT_PATH;
        filename += "/textures/";

        if (!line.empty() && fp_in.good()) {
            filename += line;
            texs.push_back(CudaTexture(filename));
        }
        return 1;
    }
    return 0;
}

int Scene::loadEnvMap()
{
    cout << "Loading Environment Map ..." << endl;
    string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    string filename = ROOT_PATH;
    filename += "/textures/";
    this->envTexId = texs.size();
    if (!line.empty() && fp_in.good()) {
        filename += line;
        texs.push_back(CudaTexture(filename));
    }
    return 1;
}

//int Scene::loadGeom(string objectid) {
//    int id = atoi(objectid.c_str());
//    if (id != geoms.size()) {
//        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
//        return -1;
//    } else {
//        cout << "Loading Geom " << id << "..." << endl;
//        Geom newGeom;
//        string line;
//
//        //load object type
//        utilityCore::safeGetline(fp_in, line);
//        if (!line.empty() && fp_in.good()) {
//            if (strcmp(line.c_str(), "sphere") == 0) {
//                cout << "Creating new sphere..." << endl;
//                newGeom.type = SPHERE;
//            } else if (strcmp(line.c_str(), "cube") == 0) {
//                cout << "Creating new cube..." << endl;
//                newGeom.type = CUBE;
//            }
//        }
//
//        //link material
//        utilityCore::safeGetline(fp_in, line);
//        if (!line.empty() && fp_in.good()) {
//            vector<string> tokens = utilityCore::tokenizeString(line);
//            newGeom.materialid = atoi(tokens[1].c_str());
//            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
//        }
//
//        //load transformations
//        utilityCore::safeGetline(fp_in, line);
//        while (!line.empty() && fp_in.good()) {
//            vector<string> tokens = utilityCore::tokenizeString(line);
//
//            //load tranformations
//            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
//                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
//            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
//                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
//            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
//                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
//            }
//
//            utilityCore::safeGetline(fp_in, line);
//        }
//
//        newGeom.transform = utilityCore::buildTransformationMatrix(
//                newGeom.translation, newGeom.rotation, newGeom.scale);
//        newGeom.inverseTransform = glm::inverse(newGeom.transform);
//        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
//
//        geoms.push_back(newGeom);
//        return 1;
//    }
//}

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
            state.maxIterations = atoi(tokens[1].c_str());
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
        string line;
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
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
            } else if (strcmp(tokens[0].c_str(), "texture") == 0) {
                newMaterial.textureId = atof(tokens[1].c_str());
                cout << "Connecting Material " << id << " to Texture " << newMaterial.textureId << "..." << endl;
            } else if (strcmp(tokens[0].c_str(), "bump") == 0) {
                newMaterial.bumpId = atof(tokens[1].c_str());
                cout << "Connecting Material " << id << " to Bump " << newMaterial.bumpId << "..." << endl;
            } else if (strcmp(tokens[0].c_str(), "SCATTER") == 0) {
                cout << "Material is scatter" << endl;
                newMaterial.isScatterMedium = true;
            }
            utilityCore::safeGetline(fp_in, line);
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
