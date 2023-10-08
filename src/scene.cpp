#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_IMPLEMENTATION
#include "scene.h"

Scene::Scene(string filename)
    : tris(), meshes(), loadedMeshGroups()
{
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
            else
            {
                vector<string> tokens = utilityCore::tokenizeString(line);
                if (strcmp(tokens[0].c_str(), "gltf") == 0)
                {
                    // this is a gltf mesh. Load it separately.
                    newGeom.type = GLTF_MESH;
                    SceneMeshGroup meshGroup = loadGltfMesh(tokens[1].c_str());
                    if (!meshGroup.valid)
                    {
                        return -1;
                    }

                    newGeom.startTriIdx = meshGroup.startTriIdx;
                    newGeom.endTriIdx = meshGroup.endTriIdx;
                    newGeom.aabbMin = meshGroup.aabbMin;
                    newGeom.aabbMax = meshGroup.aabbMax;
                }
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

SceneMeshGroup Scene::loadGltfMesh(string path)
{
    if (loadedMeshGroups.find(path) != loadedMeshGroups.end())
    {
        // Mesh already exists (might be duplicated)
        // Use the already loaded one.
        return loadedMeshGroups[path];
    }

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;

    std::string err;
    std::string warn;

    cout << "Creating new gltf scene at path " << path << "..." << endl;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty()) {
        printf("Warning: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Error: %s\n", err.c_str());
    }

    SceneMeshGroup meshGroup;
    meshGroup.valid = false;

    if (!ret) {
        cout << "Failed to parse glTF " << path << endl;
        return meshGroup;
    }

    meshGroup.startTriIdx = tris.size();
    meshGroup.startMeshIdx = meshes.size();

    int totalTris = 0;
    // Just iterate over the default scene
    for (int nodeIdx : model.scenes[model.defaultScene].nodes)
    {
        const tinygltf::Node& node = model.nodes[nodeIdx];
        totalTris += parseGltfNodeRecursive(model, node, meshGroup.aabbMin, meshGroup.aabbMax);
    }

    if (totalTris > 0)
    {
        meshGroup.endTriIdx = tris.size() - 1;
        meshGroup.endMeshIdx = meshes.size() - 1;
        meshGroup.valid = true;

        loadedMeshGroups.emplace(path, meshGroup);
    }

    return meshGroup;
}

int Scene::parseGltfNodeRecursive(const tinygltf::Model& model, const tinygltf::Node& node, glm::vec3& aabbMin, glm::vec3& aabbMax)
{
    int totalTris = 0;
    for (int childNodeIdx : node.children)
    {
        // if there are children, parse those here
        totalTris += parseGltfNodeRecursive(model, model.nodes[childNodeIdx], aabbMin, aabbMax);
    }

    totalTris += parseGltfNodeHelper(model, node, aabbMin, aabbMax);

    return totalTris;
}

/// <summary>
/// This function parses one node and all primitives inside it to build a single SceneMesh for this renderer.
/// </summary>
/// <param name="model"></param>
/// <param name="node"></param>
/// <returns></returns>
int Scene::parseGltfNodeHelper(const tinygltf::Model& model, const tinygltf::Node& node, glm::vec3& aabbMin, glm::vec3& aabbMax)
{
    int totalTris = 0;

    if (node.mesh > -1)
    {
        // has a mesh

        SceneMesh gltfMesh;
        gltfMesh.startTriIdx = tris.size();

        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const tinygltf::Primitive& prim : mesh.primitives)
        {
            auto& it = prim.attributes.find("POSITION");
            if (it != prim.attributes.end())
            {
                // get vertex positions data
                const tinygltf::Accessor& posAccessor = model.accessors[it->second];
                // use accessor to get buffer view
                const tinygltf::BufferView& posBufView = model.bufferViews[posAccessor.bufferView];
                // use buffer view to get buffer
                const tinygltf::Buffer& posBuffer = model.buffers[posBufView.buffer];
                // use posBuffer to get the positions data... finally :)
                // the smiley above is sarcastic and I should have just gone with tinyobjloader why did I chose GLTF
                const unsigned char* posDataChars = &posBuffer.data[posBufView.byteOffset + posAccessor.byteOffset];
                // oh wait I lied I still have to cast the data to a float array wups
                // we know positions are vec3s which are floats so we can simply cast it to floats
                const float* posData = reinterpret_cast<const float*>(posDataChars);
                // welp. we finally have positions. Now do this for everything else. 
                // I thought I took this class for GPU programming but here I am managing data parsing in C++

                //cout << "========= POSITIONS =========" << endl;
                int size = posBufView.byteLength / sizeof(float);
                int stride = posBufView.byteStride == 0 ? 3 : posBufView.byteStride / sizeof(float);
                int pCount = 0;
                for (int i = 0; i < size; i += stride)
                {
                    // positions are vec3s of 3 floats in GLTF 2.0 spec
                    glm::vec3 pos = glm::vec3(posData[i], posData[i + 1], posData[i + 2]);
                    gltfMesh.positions.push_back(pos);
                    //cout << "(" << gltfMesh.positions[pCount].x << "," << gltfMesh.positions[pCount].y << "," << gltfMesh.positions[pCount].z << ")" << endl;
                    pCount++;

                    gltfMesh.aabbMin = glm::min(gltfMesh.aabbMin, pos);
                    gltfMesh.aabbMax = glm::max(gltfMesh.aabbMax, pos);
                }
            }
            else
            {
                // no positions so we can't really make the mesh
                return 0;
            }

            aabbMin = glm::min(aabbMin, gltfMesh.aabbMin);
            aabbMax = glm::max(aabbMax, gltfMesh.aabbMax);

            it = prim.attributes.find("NORMAL");
            if (it != prim.attributes.end())
            {
                // get vertex positions data
                const tinygltf::Accessor& norAccessor = model.accessors[it->second];
                // use accessor to get buffer view
                const tinygltf::BufferView& norBufView = model.bufferViews[norAccessor.bufferView];
                // use buffer view to get buffer
                const tinygltf::Buffer& norBuffer = model.buffers[norBufView.buffer];
                // use posBuffer to get the positions data... finally :)
                // the smiley above is sarcastic and I should have just gone with tinyobjloader why did I chose GLTF
                const unsigned char* norDataChars = &norBuffer.data[norBufView.byteOffset + norAccessor.byteOffset];
                // oh wait I lied I still have to cast the data to a float array wups
                // we know positions are vec3s which are floats so we can simply cast it to floats
                const float* norData = reinterpret_cast<const float*>(norDataChars);
                // welp. we finally have positions. Now do this for everything else. 
                // I thought I took this class for GPU programming but here I am managing data parsing in C++

                gltfMesh.hasNormals = true;

                //cout << "========= NORMALS =========" << endl;
                int size = norBufView.byteLength / sizeof(float);
                int stride = norBufView.byteStride == 0 ? 3 : norBufView.byteStride / sizeof(float);
                int pCount = 0;
                for (int i = 0; i < size; i += stride)
                {
                    // positions are vec3s of 3 floats in GLTF 2.0 spec
                    gltfMesh.normals.push_back(glm::vec3(norData[i], norData[i + 1], norData[i + 2]));
                    //cout << "(" << gltfMesh.normals[pCount].x << "," << gltfMesh.normals[pCount].y << "," << gltfMesh.normals[pCount].z << ")" << endl;
                    pCount++;
                }
            }

            // get vertex indices data if it exists
            if (prim.indices > -1)
            {
                // these will never be interleaved so we can get them separately
                // get accessor of prim
                const tinygltf::Accessor& idxAccessor = model.accessors[prim.indices];
                // use accessor to get buffer view
                const tinygltf::BufferView& idxBufView = model.bufferViews[idxAccessor.bufferView];
                // use buffer view to get buffer
                const tinygltf::Buffer& idxBuffer = model.buffers[idxBufView.buffer];
                const unsigned char* idxDataChars = &idxBuffer.data[idxBufView.byteOffset + idxAccessor.byteOffset];
                // indices are unsigned short in GLTF 2.0 spec
                gltfMesh.indices = reinterpret_cast<const unsigned short*>(idxDataChars);
                gltfMesh.hasIndices = true;

                //cout << "========= INDICES =========" << endl;
                //for (int i = 0; i < idxAccessor.count; i++)
                //{
                //    cout << gltfMesh.indices[i] << endl;
                //}

                //cout << "========= TRIANGLE INDICES AND VTX. POS. =========" << endl;
                // Triangulate with indices
                Triangle tri;
                for (int i = 0; i < idxAccessor.count; i += 3)
                {
                    tri.v0.pos = gltfMesh.positions[gltfMesh.indices[i]];
                    tri.v1.pos = gltfMesh.positions[gltfMesh.indices[i+1]];
                    tri.v2.pos = gltfMesh.positions[gltfMesh.indices[i+2]];

                    totalTris++;

                    if (gltfMesh.hasNormals)
                    {
                        tri.v0.nor = gltfMesh.normals[gltfMesh.indices[i]];
                        tri.v1.nor = gltfMesh.normals[gltfMesh.indices[i + 1]];
                        tri.v2.nor = gltfMesh.normals[gltfMesh.indices[i + 2]];
                        tri.hasNormals = true;
                    }
                    else
                    {
                        tri.hasNormals = false;
                    }

                    tris.push_back(tri);

                    //cout << "(" << gltfMesh.indices[i] << "," << gltfMesh.indices[i+1] << "," << gltfMesh.indices[i+2] << ")" << endl;

                    //cout << "(" << gltfMesh.positions[gltfMesh.indices[i]].x << "," << gltfMesh.positions[gltfMesh.indices[i]].y << "," << gltfMesh.positions[gltfMesh.indices[i]].z << "), ("
                    //     << "(" << gltfMesh.positions[gltfMesh.indices[i+1]].x << "," << gltfMesh.positions[gltfMesh.indices[i+1]].y << "," << gltfMesh.positions[gltfMesh.indices[i+1]].z << "), ("
                    //     << "(" << gltfMesh.positions[gltfMesh.indices[i + 2]].x << "," << gltfMesh.positions[gltfMesh.indices[i + 2]].y << "," << gltfMesh.positions[gltfMesh.indices[i + 2]].z << "), (" << endl;
                }
            }
            else
            {
                // Triangulate without indices
                Triangle tri;
                for (int i = 0; i < gltfMesh.positions.size(); i += 3)
                {
                    tri.v0.pos = gltfMesh.positions[i * 3];
                    tri.v1.pos = gltfMesh.positions[i * 3 + 1];
                    tri.v2.pos = gltfMesh.positions[i * 3 + 2];

                    totalTris++;

                    if (gltfMesh.hasNormals)
                    {
                        tri.v0.nor = gltfMesh.normals[i * 3];
                        tri.v1.nor = gltfMesh.normals[i * 3 + 1];
                        tri.v2.nor = gltfMesh.normals[i * 3 + 2];
                        tri.hasNormals = true;
                    }
                    else
                    {
                        tri.hasNormals = false;
                    }
                }
            }

            // Ok mesh is loaded. Save its data for later sending to GPU
            gltfMesh.endTriIdx = gltfMesh.startTriIdx + totalTris - 1;
            meshes.push_back(gltfMesh);
        }
    }

    return totalTris;
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

    camera.apertureSize = 0.0f;
    camera.focalLength = 1.0f;

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
        } else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
            camera.apertureSize = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOCALLENGTH") == 0) {
            camera.focalLength = atof(tokens[1].c_str());
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
