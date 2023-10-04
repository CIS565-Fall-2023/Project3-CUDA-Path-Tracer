#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <tiny_gltf.h>
#include <tiny_obj_loader.h>

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    scene_filename = filename;
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
    cout << "Loading Geom " << objectid << "..." << endl;
    Geom newGeom;
    string line;

    string mesh_filename;
    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            newGeom.type = SPHERE;
        } else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            newGeom.type = CUBE;
        } else if (line.rfind("mesh", 0) == 0) {
            cout << "Loading new mesh..." << endl;
            newGeom.type = MESH;
            vector<string> tokens = utilityCore::tokenizeString(line);
            mesh_filename = tokens[1];
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        newGeom.materialid = material_map[tokens[1]];
        cout << "Connecting Geom " << objectid << " to Material " << tokens[1] << "..." << endl;
    }

    //load transformations
    for (int i = 0; i < 3; i++) {
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
    }

    newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    // TODO: load mesh
    string ext = mesh_filename.substr(mesh_filename.find_last_of(".") + 1);
    if (ext == "obj")
    {
        string scene_dirname = scene_filename.substr(0, scene_filename.find_last_of("/") + 1);
        mesh_filename = scene_dirname + mesh_filename;

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> tinyobj_materials;

        std::string err;
        bool ret = tinyobj::LoadObj(
            &attrib, &shapes, &tinyobj_materials, &err, mesh_filename.c_str(), scene_dirname.c_str(), true);
        if (!err.empty())
        {
            std::cerr << err << std::endl;
        }
        if (!ret)
        {
            std::cerr << "Failed to load/parse .obj." << std::endl;
            exit(1);
        }

        int mstartIdx = materials.size();
        // add materials
        if (tinyobj_materials.size() > 0)
        {
            for (const tinyobj::material_t& material : tinyobj_materials)
            {
                Material newMaterial;
                if (material.emission[0] + material.emission[1] + material.emission[2] > 0.0f)
                {
                    newMaterial.albedo = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
                    newMaterial.emittance = 1.0f;
                }
                else
                {
                    newMaterial.albedo = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
                    newMaterial.emittance = 0.0f;
                }
                newMaterial.metallic = material.metallic;
                newMaterial.roughness = material.roughness;
                newMaterial.ior = material.ior;
                newMaterial.opacity = material.dissolve;
                materials.push_back(newMaterial);
            }
        }

        // add vertices
        int vstartIdx = vertices.size();
        for (int i = 0; i < attrib.vertices.size() / 3; i++)
        {
            Vertex vertex;
            vertex.position = glm::vec3(newGeom.transform *
                                        glm::vec4(attrib.vertices[3 * i + 0],
                                                  attrib.vertices[3 * i + 1],
                                                  attrib.vertices[3 * i + 2], 1.0f));
            // TODO: texcoord
            // vertex.texcoord = glm::vec2(attrib.texcoords[2 * i + 0], attrib.texcoords[2 * i + 1]);
            vertices.push_back(vertex);
        }

        newGeom.meshidx = meshes.size();
        for (const tinyobj::shape_t& shape : shapes)
        {
            for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++)
            {
                Mesh newMesh;
                newMesh.v[0] = shape.mesh.indices[3 * f + 0].vertex_index + vstartIdx;
                newMesh.v[1] = shape.mesh.indices[3 * f + 1].vertex_index + vstartIdx;
                newMesh.v[2] = shape.mesh.indices[3 * f + 2].vertex_index + vstartIdx;
                newMesh.normal = glm::normalize(glm::cross(
                    vertices[newMesh.v[1]].position - vertices[newMesh.v[0]].position,
                    vertices[newMesh.v[2]].position - vertices[newMesh.v[0]].position));
                newMesh.materialid = shape.mesh.material_ids[f] < 0
                                         ? newGeom.materialid
                                         : shape.mesh.material_ids[f] + mstartIdx;

                // TODO: texcoord
                // vertices[idx0.vertex_index + vstartIdx].texcoord = glm::vec2(
                //     attrib.texcoords[2 * idx0.texcoord_index + 0], attrib.texcoords[2 * idx0.texcoord_index + 1]);
                // vertices[idx1.vertex_index + vstartIdx].texcoord = glm::vec2(
                //     attrib.texcoords[2 * idx1.texcoord_index + 0], attrib.texcoords[2 * idx1.texcoord_index + 1]);
                // vertices[idx2.vertex_index + vstartIdx].texcoord = glm::vec2(
                //     attrib.texcoords[2 * idx2.texcoord_index + 0], attrib.texcoords[2 * idx2.texcoord_index + 1]);

                meshes.push_back(newMesh);
            }
        }
        newGeom.meshcnt = meshes.size() - newGeom.meshidx;
    }

    geoms.push_back(newGeom);
    return 1;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOCALDISTANCE") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
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
    cout << "Loading Material " << materialid << "..." << endl;
    Material newMaterial;
    //load static properties
    for (int i = 0; i < 6; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RGB") == 0) {
            glm::vec3 color{atof(tokens[1].c_str()),
                            atof(tokens[2].c_str()),
                            atof(tokens[3].c_str())};
            newMaterial.albedo = glm::clamp(color, 0.0f, 1.0f);
        } else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
            newMaterial.metallic = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        } else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
            newMaterial.roughness = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        } else if (strcmp(tokens[0].c_str(), "IOR") == 0) {
            newMaterial.ior = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "OPACITY") == 0) {
            newMaterial.opacity = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
            newMaterial.emittance = atof(tokens[1].c_str());
        }
    }
    materials.push_back(newMaterial);
    material_map[materialid] = materials.size() - 1;
    return 1;
}

// ObjScene::ObjScene(string dirname, string filename)
// {
//     cout << "Reading scene from " << filename << " ..." << endl;
//     cout << " " << endl;

//     tinyobj::attrib_t attrib;
//     std::vector<tinyobj::shape_t> shapes;
//     std::vector<tinyobj::material_t> materials;

//     // get directory from filename
//     std::string err;
//     bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), dirname.c_str(), true);
//     if (!err.empty())
//     {
//         std::cerr << err << std::endl;
//     }

// }

// GLTFScene::GLTFScene(string filename)
// {

//     tinygltf::Model model;
//     tinygltf::TinyGLTF loader;
//     string warn, err;
//     bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
//     if (!warn.empty())
//     {
//         cout << "WARNING: " << warn << endl;
//     }
//     if (!err.empty())
//     {
//         cout << "ERROR: " << err << endl;
//     }
//     if (!ret)
//     {
//         cout << "Failed to parse glTF" << endl;
//         exit(1);
//     }
//     // loop through triangles
//     for (const tinygltf::Mesh& mesh : model.meshes)
//     {
//         for (const tinygltf::Primitive &primitive : mesh.primitives)
//         {
//             const tinygltf::Accessor &accessor = model.accessors[primitive.attributes.at("POSITION")];
//             const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
//             // cast to float type read only. Use accessor and bufview byte offsets to determine where position data
//             // is located in the buffer.
//             const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
//             // bufferView byteoffset + accessor byteoffset tells you where the actual position data is within the buffer. From there
//             // you should already know how the data needs to be interpreted.
//             const float *positions = reinterpret_cast<const float *>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
//             // From here, you choose what you wish to do with this position data. In this case, we  will display it out.
//             for (size_t i = 0; i < accessor.count; ++i)
//             {
//                 // Positions are Vec3 components, so for each vec3 stride, offset for x, y, and z.
//                 std::cout << "(" << positions[i * 3 + 0] << ", " // x
//                           << positions[i * 3 + 1] << ", "        // y
//                           << positions[i * 3 + 2] << ")"         // z
//                           << "\n";
//             }
//         }
//     }
//     exit(0);
// }
