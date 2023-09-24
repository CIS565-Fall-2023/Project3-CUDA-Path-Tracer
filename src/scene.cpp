#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_IMPLEMENTATION

#include "tinygltf/tiny_gltf.h"

Scene::Scene(string filename) {
    basePath = filename.substr(0, filename.rfind('/') + 1);

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
        Utils::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = Utils::tokenizeString(line);
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

int Scene::loadMesh(string filePath)
{
    if (meshIndices.find(filePath) != meshIndices.end())
    {
        return meshIndices[filePath];
    }

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, this->basePath + filePath);
    if (!warn.empty())
    {
        printf("Warn: %s\n", warn.c_str());
    }
    if (!err.empty())
    {
        printf("Err: %s\n", err.c_str());
    }
    if (!ret)
    {
        printf("Failed to parse glTF\n");
        return -1;
    }

    int startTri = tris.size();
    int numTris = 0;

    glm::vec3 minPos = glm::vec3(FLT_MAX);
    glm::vec3 maxPos = glm::vec3(-FLT_MAX);

    for (auto& nodeIndex : model.scenes[model.defaultScene].nodes)
    {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        if (node.mesh >= 0)
        {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];

            for (const tinygltf::Primitive& primitive : mesh.primitives)
            {
                const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
                const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
                const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];

                const float* positionData = reinterpret_cast<const float*>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset]);

                if (primitive.indices >= 0)
                {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                    const uint16_t* indexData = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);

                    for (size_t i = 0; i < indexAccessor.count; i += 3)
                    {
                        Triangle triangle;

                        int vertexIndex = indexData[i];
                        glm::vec3 pos = glm::vec3(positionData[vertexIndex * 3], positionData[vertexIndex * 3 + 1], positionData[vertexIndex * 3 + 2]);
                        triangle.v0 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        vertexIndex = indexData[i + 1];
                        pos = glm::vec3(positionData[vertexIndex * 3], positionData[vertexIndex * 3 + 1], positionData[vertexIndex * 3 + 2]);
                        triangle.v1 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        vertexIndex = indexData[i + 2];
                        pos = glm::vec3(positionData[vertexIndex * 3], positionData[vertexIndex * 3 + 1], positionData[vertexIndex * 3 + 2]);
                        triangle.v2 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        triangle.centroid = (triangle.v0.pos + triangle.v1.pos + triangle.v2.pos) * 0.33333333333f;

                        tris.push_back(triangle);
                        bvhTriIdx.push_back(startTri + numTris);
                        ++numTris;
                    }
                }
                else
                {
                    for (size_t i = 0; i < positionAccessor.count; i += 3)
                    {
                        Triangle triangle;
                        
                        glm::vec3 pos = glm::vec3(positionData[i * 3], positionData[i * 3 + 1], positionData[i * 3 + 2]);
                        triangle.v0 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        pos = glm::vec3(positionData[(i + 1) * 3], positionData[(i + 1) * 3 + 1], positionData[(i + 1) * 3 + 2]);
                        triangle.v1 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        pos = glm::vec3(positionData[(i + 2) * 3], positionData[(i + 2) * 3 + 1], positionData[(i + 2) * 3 + 2]);
                        triangle.v2 = { pos };
                        minPos = glm::min(minPos, pos);
                        maxPos = glm::max(maxPos, pos);

                        triangle.centroid = (triangle.v0.pos + triangle.v1.pos + triangle.v2.pos) * 0.33333333333f;

                        tris.push_back(triangle);
                        bvhTriIdx.push_back(startTri + numTris);
                        ++numTris;
                    }
                }
            }
        }
    }

    Mesh newMesh;
    newMesh.bvhRootNode = buildBvh(startTri, numTris);

    int meshIndex = meshes.size();
    meshes.push_back(newMesh);
    meshIndices[filePath] = meshIndex;
    return meshIndex;
}

int Scene::buildBvh(int startTri, int numTris)
{
    bvhNodes.reserve(bvhNodes.size() + 2 * numTris);

    int rootNodeIdx = bvhNodes.size();
    bvhNodes.emplace_back();
    BvhNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = startTri, root.triCount = numTris;
    bvhUpdateNodeBounds(root);
    bvhSubdivide(root);

    /*
    int totalTris = 0;
    for (int i = 0; i < bvhNodes.size(); ++i)
    {
        cout << bvhNodes[i] << endl;
        cout << endl;
        totalTris += bvhNodes[i].triCount;
    }

    cout << numTris << endl;
    cout << totalTris << endl;
    */

    return rootNodeIdx;
}

void Scene::bvhUpdateNodeBounds(BvhNode& node)
{
    node.aabbMin = glm::vec3(FLT_MAX);
    node.aabbMax = glm::vec3(-FLT_MAX);
    for (int i = 0; i < node.triCount; ++i)
    {
        Triangle& leafTri = tris[bvhTriIdx[node.leftFirst + i]];
        node.aabbMin = glm::min(node.aabbMin, leafTri.v0.pos);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v1.pos);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v2.pos);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v0.pos);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v1.pos);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v2.pos);
    }
}

void Scene::bvhSubdivide(BvhNode& node)
{
    if (node.triCount <= 2)
    {
        return;
    }

    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x)
    {
        axis = 1;
    }
    if (extent.z > extent[axis])
    {
        axis = 2;
    }
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

    int i = node.leftFirst;
    int j = i + node.triCount - 1;
    while (i <= j)
    {
        if (tris[bvhTriIdx[i]].centroid[axis] < splitPos)
        {
            ++i;
        }
        else
        {
            std::swap(bvhTriIdx[i], bvhTriIdx[j--]);
        }
    }

    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount)
    {
        return;
    }

    int leftChildIdx = bvhNodes.size();

    bvhNodes.emplace_back();
    BvhNode& leftChild = bvhNodes.back();
    leftChild.leftFirst = node.leftFirst;
    leftChild.triCount = leftCount;

    bvhNodes.emplace_back();
    BvhNode& rightChild = bvhNodes.back();
    rightChild.leftFirst = i;
    rightChild.triCount = node.triCount - leftCount;

    node.leftFirst = leftChildIdx;
    node.triCount = 0;

    bvhUpdateNodeBounds(leftChild);
    bvhUpdateNodeBounds(rightChild);

    bvhSubdivide(leftChild);
    bvhSubdivide(rightChild);
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
        Utils::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), "mesh") == 0) {
                string filePath;
                Utils::safeGetline(fp_in, filePath);
                string fullPath = basePath + filePath;
                cout << "Creating new mesh from " << fullPath << "..." << endl;
                newGeom.type = MESH;

                auto time1 = Utils::timeSinceEpochMillisec();
                newGeom.referenceId = loadMesh(fullPath);
                auto time2 = Utils::timeSinceEpochMillisec();
                auto timeTaken = (time2 - time1);
                cout << "Took " << timeTaken << " ms to build BVH" << endl;
            }
        }

        //link material
        Utils::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = Utils::tokenizeString(line);
            newGeom.materialId = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialId << "..." << endl;
        }

        //load transformations
        Utils::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = Utils::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            Utils::safeGetline(fp_in, line);
        }

        newGeom.transform = Utils::buildTransformationMatrix(
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
        Utils::safeGetline(fp_in, line);
        vector<string> tokens = Utils::tokenizeString(line);
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

#if DEBUG_SHOW_NORMALS
    state.traceDepth = 1;
#endif

    string line;
    Utils::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = Utils::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOCAL_DIST") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
        }

        Utils::safeGetline(fp_in, line);
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

int Scene::loadMaterial(string materialId) {
    int id = atoi(materialId.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            Utils::safeGetline(fp_in, line);
            vector<string> tokens = Utils::tokenizeString(line);
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
