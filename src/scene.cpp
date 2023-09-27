#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

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
                int loadObjRes = loadGeom(tokens[1]);
                objCount += loadObjRes;
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
    if (id != objCount) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        int triIdx = 0;
        newGeom.scale = glm::vec3(1.0f);
        string meshPath;
        Triangle singleTri;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), "triangle") == 0) {
                cout << "Creating new triangle..." << endl;
                newGeom.type = TRIANGLE;
            } else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations & vertices
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
            } else if (strcmp(tokens[0].c_str(), "VERTEX") == 0 && newGeom.type == TRIANGLE && triIdx < 3) {
                singleTri.pos[triIdx] = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                triIdx++;
            } else if (strcmp(tokens[0].c_str(), "MESH_PATH") == 0 && newGeom.type == MESH) {
                meshPath = tokens[1];
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // load triangle
        if (newGeom.type == TRIANGLE) {
            if (triIdx < 3) {
                std::cout << "ERROR: triangle lose point!!!" << std::endl;
            }
            for (int i = 0; i < 3; i++) {
                glm::vec4 newPos = newGeom.transform * glm::vec4(singleTri.pos[i][0], singleTri.pos[i][1], singleTri.pos[i][2], 1.0f);
                singleTri.pos[i] = glm::vec3(newPos.x, newPos.y, newPos.z);
            }
            singleTri.materialid = newGeom.materialid;
            tris.push_back(singleTri);
        }
        // load mesh
        else if (newGeom.type == MESH) {
            if (meshPath.empty()) {
                std::cout << "ERROR: mesh has no path!!!" << std::endl;
                return 0;
            }
            std::vector<tinyobj::shape_t> shapes; std::vector<tinyobj::material_t> materials;
            std::string errors = tinyobj::LoadObj(shapes, materials, meshPath.c_str());
            std::cout << errors << std::endl;
            if (errors.size() == 0)
            {
                //Read the information from the vector of shape_ts
                for (unsigned int i = 0; i < shapes.size(); i++)
                {
                    std::vector<float>& positions = shapes[i].mesh.positions;
                    std::vector<float>& normals = shapes[i].mesh.normals;
                    std::vector<float>& uvs = shapes[i].mesh.texcoords;
                    std::vector<unsigned int>& indices = shapes[i].mesh.indices;
                    for (unsigned int j = 0; j < indices.size(); j += 3)
                    {
                        Triangle t;
                        t.pos[0] = glm::vec3(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
                        t.pos[1] = glm::vec3(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
                        t.pos[2] = glm::vec3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);
                        //if (normals.size() > 0)
                        //{
                        //    t.nor[0] = glm::vec3 (normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
                        //    t.nor[1] = glm::vec3 (normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
                        //    t.nor[2] = glm::vec3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
                        //}
                        if (uvs.size() > 0)
                        {
                            t.uv[0] = glm::vec2(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1]);
                            t.uv[1] = glm::vec2 (uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1]);
                            t.uv[2] = glm::vec2 (uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1]);
                        }
                        for (int k = 0; k < 3; k++) {
                            glm::vec4 newPos = newGeom.transform * glm::vec4(t.pos[k][0], t.pos[k][1], t.pos[k][2], 1.0f);
                            t.pos[k] = glm::vec3(newPos.x, newPos.y, newPos.z);
                        }
                        t.materialid = newGeom.materialid;
                        tris.push_back(t);
                    }
                    std::cout << "shape " << i << " has " << indices.size() << " triangles" << std::endl;
                }
                
            }
        }
        else { // load other objects
            geoms.push_back(newGeom);
        }


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
