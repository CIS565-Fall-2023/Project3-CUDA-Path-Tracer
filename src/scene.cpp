#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

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

int Scene::loadMesh(const string& fp, int& primStartIdx, int& primCnt) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    string err;
    string warn;
    bool ret;

    // Read from file
    if (utilityCore::matchFileExtension(fp, "gltf")) {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, fp);
    }
    else if (utilityCore::matchFileExtension(fp, "glb")) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, fp);
    }
    else {
        cout << "Reading " << fp << "failed! Unsupported file type." << endl;
        exit(-1);
    }
    if (!warn.empty()) {
        cout << "TinyGLTF Warnings: " << warn << endl;
    }
    if (!err.empty()) {
        cout << "TinyGLTF Errors: " << err << endl;
    }
    if (!ret) {
        exit(-1);
    }
    cout << "Successfully read file. Processing..." << endl;

    // Load materials to memory
    int matStartIdx = materials.size();
    for (const tinygltf::Material& material : model.materials) {
        const int textureIdx = material.pbrMetallicRoughness.baseColorTexture.index;
        if (textureIdx < 0) {
            continue;
        }
        const tinygltf::Image& img = model.images[model.textures[textureIdx].source];
        textures.emplace_back();
        Texture& tex = textures.back();
        tex.host_buffer = new float[img.width * img.height * 4];
        for (int i = 0; i < img.image.size(); i++) {
            tex.host_buffer[i] = img.image[i] / 255.f;
        }
        tex.width = img.width;
        tex.height = img.height;
        tex.channels = 4;
        materials.emplace_back();
        Material& mat = materials.back();
        mat.textureid = textures.size() - 1;
        mat.color = glm::vec3(1.f, 0.f, 0.f);
        mat.emittance = 0;
        mat.hasReflective = 0;
        mat.hasRefractive = 0;
        mat.indexOfRefraction = 0;
        mat.specular.color = glm::vec3(0.f);
        mat.specular.exponent = 0;
    }
    
    // Load primitives
    primStartIdx = prims.size();
    primCnt = 0;
    for (const tinygltf::Mesh& mesh : model.meshes) {
        for (const tinygltf::Primitive& primitive : mesh.primitives) {
            const int primMatId = primitive.material >= 0 ? matStartIdx + primitive.material : -1;

            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            const float* posArray = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

            const float* norArray = nullptr;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                const tinygltf::Accessor& norAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
                const tinygltf::Buffer& norBuffer = model.buffers[norBufferView.buffer];
                norArray = reinterpret_cast<const float*>(&norBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]);
            }
            
            const float* uvArray = nullptr;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const tinygltf::BufferView& uvBufferView = model.bufferViews[uvAccessor.bufferView];
                const tinygltf::Buffer& uvBuffer = model.buffers[uvBufferView.buffer];
                uvArray = reinterpret_cast<const float*>(&uvBuffer.data[uvBufferView.byteOffset + uvAccessor.byteOffset]);
            }

            if (primitive.indices < 0) {
                // vertices are not shared (not indexed)
                for (size_t i = 0; i < posAccessor.count; i += 3) {
                    Triangle triangle;
                    triangle.v1.pos = glm::vec3(posArray[i * 3], posArray[i * 3 + 1], posArray[i * 3 + 2]);
                    triangle.v2.pos = glm::vec3(posArray[(i + 1) * 3], posArray[(i + 1) * 3 + 1], posArray[(i + 1) * 3 + 2]);
                    triangle.v3.pos = glm::vec3(posArray[(i + 2) * 3], posArray[(i + 2) * 3 + 1], posArray[(i + 2) * 3 + 2]);

                    if (norArray) {
                        triangle.v1.nor = glm::vec3(norArray[i * 3], norArray[i * 3 + 1], norArray[i * 3 + 2]);
                        triangle.v2.nor = glm::vec3(norArray[(i + 1) * 3], norArray[(i + 1) * 3 + 1], norArray[(i + 1) * 3 + 2]);
                        triangle.v3.nor = glm::vec3(norArray[(i + 2) * 3], norArray[(i + 2) * 3 + 1], norArray[(i + 2) * 3 + 2]);
                    }

                    if (uvArray) {
                        triangle.v1.uv = glm::vec2(uvArray[i * 2], uvArray[i * 2 + 1]);
                        triangle.v2.uv = glm::vec2(uvArray[(i + 1) * 2], uvArray[(i + 1) * 2 + 1]);
                        triangle.v3.uv = glm::vec2(uvArray[(i + 2) * 2], uvArray[(i + 2) * 2 + 1]);
                    }

                    triangle.centroid = (triangle.v1.pos + triangle.v2.pos + triangle.v3.pos) * 0.33333333333f;
                    triangle.materialid = primMatId;
                    prims.push_back(triangle);
                    ++primCnt;
                }
            } else {
                const tinygltf::Accessor& indAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indBufferView = model.bufferViews[indAccessor.bufferView];
                const tinygltf::Buffer& indBuffer = model.buffers[indBufferView.buffer];

                const uint16_t* indArray = reinterpret_cast<const uint16_t*>(&indBuffer.data[indBufferView.byteOffset + indAccessor.byteOffset]);
                for (size_t i = 0; i < indAccessor.count; i += 3) {
                    Triangle triangle;

                    const int v1 = indArray[i];
                    const int v2 = indArray[i + 1];
                    const int v3 = indArray[i + 2];

                    triangle.v1.pos = glm::vec3(posArray[v1 * 3], posArray[v1 * 3 + 1], posArray[v1 * 3 + 2]);
                    triangle.v2.pos = glm::vec3(posArray[v2 * 3], posArray[v2 * 3 + 1], posArray[v2 * 3 + 2]);
                    triangle.v3.pos = glm::vec3(posArray[v3 * 3], posArray[v3 * 3 + 1], posArray[v3 * 3 + 2]);

                    if (norArray) {
                        triangle.v1.nor = glm::vec3(norArray[v1 * 3], norArray[v1 * 3 + 1], norArray[v1 * 3 + 2]);
                        triangle.v2.nor = glm::vec3(norArray[v2 * 3], norArray[v2 * 3 + 1], norArray[v2 * 3 + 2]);
                        triangle.v3.nor = glm::vec3(norArray[v3 * 3], norArray[v3 * 3 + 1], norArray[v3 * 3 + 2]);
                    }

                    if (uvArray) {
                        triangle.v1.uv = glm::vec2(uvArray[v1 * 2], uvArray[v1 * 2 + 1]);
                        triangle.v2.uv = glm::vec2(uvArray[v2 * 2], uvArray[v2 * 2 + 1]);
                        triangle.v3.uv = glm::vec2(uvArray[v3 * 2], uvArray[v3 * 2 + 1]);
                    }

                    triangle.centroid = (triangle.v1.pos + triangle.v2.pos + triangle.v3.pos) * 0.33333333333f;
                    triangle.materialid = primMatId;
                    prims.push_back(triangle);
                    ++primCnt;
                }
            }
        }
    }

    return buildBVH(primStartIdx, primCnt);
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
            else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Reading Mesh..." << endl;
                string meshFp;
                utilityCore::safeGetline(fp_in, meshFp);
                newGeom.type = MESH;
                newGeom.bvhRootIdx = loadMesh(meshFp, newGeom.primStartIdx, newGeom.primCnt);
                cout << "Loaded " << utilityCore::extractFilename(meshFp) << " with " << newGeom.primCnt << " triangles." << endl;
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
        } else if (strcmp(tokens[0].c_str(), "FOCALDIST") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
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

int Scene::buildBVH(int startPrim, int numPrim) {
    // a BVH for a mesh is at most 2N - 1 nodes, where N is number of triangles in mesh
    BVHNodes.reserve(BVHNodes.size() + 2 * numPrim);
    
    int rootIdx = BVHNodes.size();
    BVHNodes.emplace_back(startPrim, numPrim);
    BVHNode& root = BVHNodes[rootIdx];
    bvhUpdateBounds(root);
    bvhSubdivide(root);
    return rootIdx;
}

void Scene::bvhUpdateBounds(BVHNode& node) {
    node.bounds = AABoundBox();
    for (int i = 0; i < node.primCnt; i++) {
        const Triangle& t = prims[node.leftInd + i];
        node.bounds.grow(t);
    }
}

float Scene::bvhBestSplitPlane(BVHNode& node, int& axis, float& splitPos, AABoundBox& leftChild, AABoundBox& rightChild) {
    float bestCost = FLT_MAX;
    for (int testAxis = 0; testAxis < 3; testAxis++) {
        float boundsMin = FLT_MAX;
        float boundsMax = -FLT_MAX;
        for (int i = 0; i < node.primCnt; i++) {
            const Triangle& triangle = prims[node.leftInd + i];
            boundsMin = min(boundsMin, triangle.centroid[testAxis]);
            boundsMax = max(boundsMax, triangle.centroid[testAxis]);
        }
        if (boundsMin == boundsMax) continue;
        // populate the bins
        BVHBin bin[NUM_BVHBINS];
        float scale = NUM_BVHBINS / (boundsMax - boundsMin);
        for (int i = 0; i < node.primCnt; i++) {
            const Triangle& triangle = prims[node.leftInd + i];
            int binIdx = min(NUM_BVHBINS - 1, (int)((triangle.centroid[testAxis] - boundsMin) * scale));
            bin[binIdx].primCnt++;
            bin[binIdx].bounds.grow(triangle);
        }
        // gather data for the planes between the NUM_BVHBINS bins
        AABoundBox leftBoxes[NUM_BVHBINS - 1], rightBoxes[NUM_BVHBINS - 1];
        int leftCount[NUM_BVHBINS - 1], rightCount[NUM_BVHBINS - 1];
        AABoundBox leftBox, rightBox;
        int leftSum = 0, rightSum = 0;
        for (int i = 0; i < NUM_BVHBINS - 1; i++) {
            leftSum += bin[i].primCnt;
            leftCount[i] = leftSum;
            leftBox.grow(bin[i].bounds);
            leftBoxes[i] = leftBox;
            rightSum += bin[NUM_BVHBINS - 1 - i].primCnt;
            rightCount[NUM_BVHBINS - 2 - i] = rightSum;
            rightBox.grow(bin[NUM_BVHBINS - 1 - i].bounds);
            rightBoxes[NUM_BVHBINS - 2 - i] = rightBox;
        }
        // calculate SAH cost for the NUM_BVHBINS planes
        scale = (boundsMax - boundsMin) / NUM_BVHBINS;
        for (int i = 0; i < NUM_BVHBINS - 1; i++) {
            float planeCost = leftCount[i] * leftBoxes[i].surfaceArea() + rightCount[i] * rightBoxes[i].surfaceArea();
            if (planeCost < bestCost) {
                axis = testAxis;
                splitPos = boundsMin + scale * (i + 1);
                leftChild = leftBoxes[i];
                rightChild = rightBoxes[i];
                bestCost = planeCost;
            }
        }
    }
    return bestCost;
}

void Scene::bvhSubdivide(BVHNode& node) {
    // determine split axis and position
    int axis = 0;
    float splitPos;
    AABoundBox leftChildBox, rightChildBox;
    float splitCost = bvhBestSplitPlane(node, axis, splitPos, leftChildBox, rightChildBox);
    if (splitCost >= node.scanCost()) {
        return;
    }
    // partition by split position
    int i = node.leftInd;
    int j = i + node.primCnt - 1;
    while (i <= j) {
        if (prims[i].centroid[axis] < splitPos) {
            i++;
        }
        else {
            std::swap(prims[i], prims[j--]);
        }
    }

    // abort split if one of the sides is empty
    int leftCount = i - node.leftInd;
    if (leftCount == 0 || leftCount == node.primCnt) return;

    // create child nodes
    int leftChildIdx = BVHNodes.size(); // one greater than last node in vector
    BVHNodes.emplace_back(leftChildBox, node.leftInd, leftCount); // construct leftChild directly in vector
    int rightChildIdx = BVHNodes.size(); // one greater than last node (left child) in vector
    BVHNodes.emplace_back(rightChildBox, i, node.primCnt - leftCount);
    
    node.leftInd = leftChildIdx;
    node.primCnt = 0;

    // recursively build child nodes
    bvhSubdivide(BVHNodes[leftChildIdx]);
    bvhSubdivide(BVHNodes[rightChildIdx]);
}
