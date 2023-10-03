#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/intersect.hpp>

#define TINYGLTF_IMPLEMENTATION

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

// from tinygltf example basic
// recursively iter through scene nodes to parse meshes and associated maps
void Scene::parseMesh(tinygltf::Model& model, tinygltf::Mesh& mesh, std::vector<Geom>& newGeoms) {
    //1 Geom per prim(submesh)
    for (const tinygltf::Primitive& prim : mesh.primitives) {
        //standardized data structs
        glm::vec3 bb_min(FLT_MAX), bb_max(FLT_MIN);
        Geom mesh_geom;
        mesh_geom.type = MESH_PRIM;
        std::vector<Triangle> curr_tris;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<int> curr_indices;
        std::vector<glm::vec2> uvs;
        std::vector<MeshPoint> curr_mesh_pts;
        for (const std::pair<const std::string, int>& attribute : prim.attributes) {
            //locate corresponding accessor, buf view, buf
            const tinygltf::Accessor& accessor = model.accessors[attribute.second];
            const tinygltf::BufferView& buf_view = model.bufferViews[accessor.bufferView];
            tinygltf::Buffer& buf = model.buffers[buf_view.buffer];

            // offset to where data begins for this attribute
            unsigned char* attrib_data = buf.data.data() + buf_view.byteOffset;

            if (attribute.first == "POSITION") {
                //assuming float vec3s bc im lazy FIXME
                glm::vec3* casted_ad = (glm::vec3*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    vertices.push_back(casted_ad[i]);
                }

                //aggregate bb min and max for mesh across prims
                bb_min = glm::min(bb_min, glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]));
                bb_max = glm::max(bb_max, glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]));
            }
            else if (attribute.first == "NORMAL") {
                //assuming float vec3
                glm::vec3* casted_ad = (glm::vec3*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    normals.push_back(casted_ad[i]);
                }
            }
            //1 set of texture coords per mesh FIXME
            //to fix need to look at refactor structs for more uvs
            else if (attribute.first == "TEXCOORD_0") {
                //assuming vec2 float
                glm::vec2* tex = (glm::vec2*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    uvs.push_back(tex[i]);
                }
            }
        }
        
        //tri indices parse
        const tinygltf::Accessor& indices_acc = model.accessors[prim.indices];
        const tinygltf::BufferView& buf_view = model.bufferViews[indices_acc.bufferView];
        tinygltf::Buffer& buf = model.buffers[buf_view.buffer];
        unsigned char* indices_data = buf.data.data() + buf_view.byteOffset;

        //assuming unsigned short indices bc lazy FIXME
        unsigned short* casted_indices = (unsigned short*)indices_data;
        for (int i = 0; i < indices_acc.count; i++) {
            curr_indices.push_back((int)(casted_indices[i]));
        }

        for (int i = 0; i < vertices.size(); i++) {
            //defaults if no norm/uv for mesh
            glm::vec3 norm = normals.size() > i ? normals[i] : glm::vec3(0);
            glm::vec2 uv = uvs.size() > i ? uvs[i] : glm::vec2(-1);
            MeshPoint mp = { vertices[i], norm, uv };
            curr_mesh_pts.push_back(mp);
        }

        for (int i = 0; i < curr_indices.size() / 3; i++) {
            Triangle tri = { curr_mesh_pts[curr_indices[(3 * i)]], curr_mesh_pts[curr_indices[(3 * i) + 1]] , curr_mesh_pts[curr_indices[(3 * i) + 2]] };
            curr_tris.push_back(tri);
        }

        mesh_geom.tri_start_index = mesh_triangles.size();
        mesh_triangles.insert(mesh_triangles.end(), curr_tris.begin(), curr_tris.end());
        mesh_geom.tri_end_index = mesh_triangles.size();

        mesh_geom.bb_min = bb_min;
        mesh_geom.bb_max = bb_max;

        // load mesh texture data + texture if not already loaded
        tinygltf::Material &prim_mat = model.materials[prim.material];
        tinygltf::TextureInfo& color_tex = prim_mat.pbrMetallicRoughness.baseColorTexture;
        tinygltf::NormalTextureInfo& normal_tex = prim_mat.normalTexture;

        //base color tex
        if (color_tex.index >= 0) {
            tinygltf::Image img = model.images[model.textures[color_tex.index].source];
            //if tex already loaded link it
            if (imguri_to_index.find(img.uri) != imguri_to_index.end()) {
                mesh_geom.texture_index = imguri_to_index[img.uri];
            }
            // if not load it
            else {
                int bytes_p_channel = img.bits / 8;
                std::vector<glm::vec3> rgb_data;
                //img data is 4 channel rgba?
                if (bytes_p_channel == 1) {
                    for (int i = 0; i < img.image.size(); i += 4) {
                        //each byte is 1 channel
                        rgb_data.push_back(glm::vec3((short)(img.image[i]) / 255.f, (short)(img.image[i + 1]) / 255.f, short(img.image[i + 2]) / 255.f));
                    }
                } else if (bytes_p_channel == 2) {
                    unsigned short* casted_2b = (unsigned short*)img.image.data();
                    for (int i = 0; i < img.image.size() / 2; i += 4) {
                        //each byte is 1 channel
                        rgb_data.push_back(glm::vec3(casted_2b[i] / 65535.f, casted_2b[i + 1] / 65535.f, casted_2b[i + 2] / 65535.f));
                    }
                } else if (bytes_p_channel == 4) {
                    unsigned int* casted_4b = (unsigned int*)img.image.data();
                    for (int i = 0; i < img.image.size() / 4; i += 4) {
                        //each byte is 1 channel
                        rgb_data.push_back(glm::vec3(casted_4b[i] / 4294967295.f, casted_4b[i + 1] / 4294967295.f, casted_4b[i + 2] / 4294967295.f));
                    }
                }
                // add image info and 
                mesh_geom.texture_index = image_infos.size();
                imguri_to_index.insert({ img.uri, image_infos.size() });
                ImageInfo img_info = { image_data.size(), img.width, img.height };
                image_infos.push_back(img_info);

                image_data.insert(image_data.end(), rgb_data.begin(), rgb_data.end());
            }
        }

        // normal map img 
        if (normal_tex.index >= 0) {
            tinygltf::Image img = model.images[model.textures[normal_tex.index].source];
            //if tex already loaded link it
            if (imguri_to_index.find(img.uri) != imguri_to_index.end()) {
                mesh_geom.texture_index = imguri_to_index[img.uri];
            }
            // if not load it
            else {
                int bytes_p_channel = img.bits / 8;
                std::vector<glm::vec3> normal_data;
                //img data is 4 channel rgba?
                if (bytes_p_channel == 1) {
                    for (int i = 0; i < img.image.size(); i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(img.image[i] / 255.f, img.image[i + 1] / 255.f, img.image[i + 2] / 255.f);
                        // to convert to normals (-0.5), (*2)
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                }
                else if (bytes_p_channel == 2) {
                    unsigned short* casted_2b = (unsigned short*)img.image.data();
                    for (int i = 0; i < img.image.size() / 2; i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(casted_2b[i] / 65535.f, casted_2b[i + 1] / 65535.f, casted_2b[i + 2] / 65535.f);
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                } else if (bytes_p_channel == 4) {
                    unsigned int* casted_4b = (unsigned int*)img.image.data();
                    for (int i = 0; i < img.image.size() / 4; i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(casted_4b[i] / 4294967295.f, casted_4b[i + 1] / 4294967295.f, casted_4b[i + 2] / 4294967295.f);
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                }
                // add image info and 
                mesh_geom.normal_map_index = image_infos.size();
                imguri_to_index.insert({ img.name, image_infos.size() });
                ImageInfo img_info = { image_data.size(), img.width, img.height };
                image_infos.push_back(img_info);

                image_data.insert(image_data.end(), normal_data.begin(), normal_data.end());
            }
        }

        cout << "Created new mesh: " << mesh.name << endl;
        newGeoms.push_back(mesh_geom);
    }
}

void Scene::parseModelNodes(tinygltf::Model& model, tinygltf::Node& node, std::vector<Geom>& newGeoms) {

    if (node.mesh >= 0 && node.mesh < model.meshes.size()) {
        parseMesh(model, model.meshes[node.mesh], newGeoms);
    } 

    for (size_t i = 0; i < node.children.size(); i++) {
        parseModelNodes(model, model.nodes[node.children[i]], newGeoms);
    }
}

int Scene::loadGltf(string file, std::vector<Geom> &newGeoms) {
    //gltf loader and model handle + fn
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, file);
    if (!warn.empty() || !err.empty() || !ret) {
        cout << warn << ", " << err << endl;
        cout << "failed to read gltf file" << endl;
        return -1;
    }

    //one gltf file can produce many Geom mesh objs
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    //iter through all scene nodes for meshes
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        parseModelNodes(model, model.nodes[scene.nodes[i]], newGeoms);
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    cout << "Loading Geom " << id << "..." << endl;
    std::vector<Geom> newGeoms;
    string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            Geom newSphere;
            newSphere.type = SPHERE;
            newGeoms.push_back(newSphere);
        } else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            Geom newCube;
            newCube.type = CUBE;
            newGeoms.push_back(newCube);
        // if file ext of str is .gltf load model with tinygltf
        } else if (line.find('.') != std::string::npos && line.substr(line.find_last_of('.') + 1) == "gltf") {
            loadGltf(line, newGeoms);
        } else {
            // bad input
            cerr << "Bad input object :(" << endl;
            exit(-1);
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        for (auto& geom : newGeoms) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            geom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << geom.materialid << "..." << endl;
        }
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            for (auto& geom : newGeoms) {
                geom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            for (auto& geom : newGeoms) {
                geom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            for (auto& geom : newGeoms) {
                geom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        }

        utilityCore::safeGetline(fp_in, line);
    }

    for (auto& geom : newGeoms) {
        geom.transform = utilityCore::buildTransformationMatrix(
            geom.translation, geom.rotation, geom.scale);
        geom.inverseTransform = glm::inverse(geom.transform);
        geom.invTranspose = glm::inverseTranspose(geom.transform);
    }

    geoms.insert(geoms.end(), newGeoms.begin(), newGeoms.end());
    return 1;
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
