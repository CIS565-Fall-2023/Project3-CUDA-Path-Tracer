#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "utilities.h"

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
                cout << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << endl;
            }
        }
    }
}

Scene::~Scene()
{
    for (const auto texture : textures)
    {
        delete[] texture.host_dataPtr;
    }
}

int Scene::loadMesh(string filePath)
{
    if (bvhRootIndices.find(filePath) != bvhRootIndices.end())
    {
        return bvhRootIndices[filePath];
    }

    cout << "Loading mesh: " << filePath << endl;

    auto time1 = Utils::timeSinceEpochMillisec();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret;

    if (Utils::filePathHasExtension(filePath, ".glb"))
    {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, this->basePath + filePath);
    }
    else if (Utils::filePathHasExtension(filePath, ".gltf"))
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, this->basePath + filePath);
    }
    else
    {
        printf("File extension not supported\n");
        return -1;
    }

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

    auto time2 = Utils::timeSinceEpochMillisec();
    auto timeTaken = (time2 - time1);
    cout << "Took " << timeTaken << " ms to read mesh file" << endl;

    time1 = Utils::timeSinceEpochMillisec();

    int startTri = tris.size();
    int numTris = 0;

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

                const float* normalData = nullptr;
                if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
                {
                    const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                    const tinygltf::BufferView& normalBufferView = model.bufferViews[normalAccessor.bufferView];
                    const tinygltf::Buffer& normalBuffer = model.buffers[normalBufferView.buffer];
                    normalData = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset]);
                }

                const float* uvData = nullptr;
                if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
                {
                    const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                    const tinygltf::BufferView& uvBufferView = model.bufferViews[uvAccessor.bufferView];
                    const tinygltf::Buffer& uvBuffer = model.buffers[uvBufferView.buffer];
                    uvData = reinterpret_cast<const float*>(&uvBuffer.data[uvBufferView.byteOffset + uvAccessor.byteOffset]);
                }

                if (primitive.indices >= 0)
                {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                    const uint16_t* indexData = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);

                    for (size_t i = 0; i < indexAccessor.count; i += 3)
                    {
                        Triangle triangle;

                        const int vIdx0 = indexData[i];
                        const int vIdx1 = indexData[i + 1];
                        const int vIdx2 = indexData[i + 2];

                        triangle.v0.pos = glm::vec3(positionData[vIdx0 * 3], positionData[vIdx0 * 3 + 1], positionData[vIdx0 * 3 + 2]);
                        triangle.v1.pos = glm::vec3(positionData[vIdx1 * 3], positionData[vIdx1 * 3 + 1], positionData[vIdx1 * 3 + 2]);
                        triangle.v2.pos = glm::vec3(positionData[vIdx2 * 3], positionData[vIdx2 * 3 + 1], positionData[vIdx2 * 3 + 2]);

                        if (normalData)
                        {
                            triangle.v0.nor = glm::vec3(normalData[vIdx0 * 3], normalData[vIdx0 * 3 + 1], normalData[vIdx0 * 3 + 2]);
                            triangle.v1.nor = glm::vec3(normalData[vIdx1 * 3], normalData[vIdx1 * 3 + 1], normalData[vIdx1 * 3 + 2]);
                            triangle.v2.nor = glm::vec3(normalData[vIdx2 * 3], normalData[vIdx2 * 3 + 1], normalData[vIdx2 * 3 + 2]);
                        }

                        if (uvData)
                        {
                            triangle.v0.uv = glm::vec2(uvData[vIdx0 * 2], uvData[vIdx0 * 2 + 1]);
                            triangle.v1.uv = glm::vec2(uvData[vIdx1 * 2], uvData[vIdx1 * 2 + 1]);
                            triangle.v2.uv = glm::vec2(uvData[vIdx2 * 2], uvData[vIdx2 * 2 + 1]);
                        }

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

                        triangle.v0.pos = glm::vec3(positionData[i * 3], positionData[i * 3 + 1], positionData[i * 3 + 2]);
                        triangle.v1.pos = glm::vec3(positionData[(i + 1) * 3], positionData[(i + 1) * 3 + 1], positionData[(i + 1) * 3 + 2]);
                        triangle.v2.pos = glm::vec3(positionData[(i + 2) * 3], positionData[(i + 2) * 3 + 1], positionData[(i + 2) * 3 + 2]);

                        if (normalData)
                        {
                            triangle.v0.nor = glm::vec3(normalData[i * 3], normalData[i * 3 + 1], normalData[i * 3 + 2]);
                            triangle.v1.nor = glm::vec3(normalData[(i + 1) * 3], normalData[(i + 1) * 3 + 1], normalData[(i + 1) * 3 + 2]);
                            triangle.v2.nor = glm::vec3(normalData[(i + 2) * 3], normalData[(i + 2) * 3 + 1], normalData[(i + 2) * 3 + 2]);
                        }

                        if (uvData)
                        {
                            triangle.v0.uv = glm::vec2(uvData[i * 2], uvData[i * 2 + 1]);
                            triangle.v1.uv = glm::vec2(uvData[(i + 1) * 2], uvData[(i + 1) * 2 + 1]);
                            triangle.v2.uv = glm::vec2(uvData[(i + 2) * 2], uvData[(i + 2) * 2 + 1]);
                        }

                        triangle.centroid = (triangle.v0.pos + triangle.v1.pos + triangle.v2.pos) * 0.33333333333f;

                        tris.push_back(triangle);
                        bvhTriIdx.push_back(startTri + numTris);
                        ++numTris;
                    }
                }
            }
        }
    }

    time2 = Utils::timeSinceEpochMillisec();
    timeTaken = (time2 - time1);
    cout << "Took " << timeTaken << " ms to populate mesh triangles" << endl;

    int bvhRootNodeIdx = buildBvh(startTri, numTris);
    bvhRootIndices[filePath] = bvhRootNodeIdx;
    return bvhRootNodeIdx;
}

int Scene::buildBvh(int startTri, int numTris)
{
    auto time1 = Utils::timeSinceEpochMillisec();

    bvhNodes.reserve(bvhNodes.size() + 2 * numTris);

    int rootNodeIdx = bvhNodes.size();
    bvhNodes.emplace_back();
    BvhNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = startTri, root.triCount = numTris;
    bvhUpdateNodeBounds(root);
    bvhSubdivide(root);

    auto time2 = Utils::timeSinceEpochMillisec();
    auto timeTaken = (time2 - time1);
    cout << "Took " << timeTaken << " ms to build BVH" << endl;

#if DEBUG_PRINT_BVH
    int totalTris = 0;
    for (int i = 0; i < bvhNodes.size(); ++i)
    {
        cout << bvhNodes[i] << endl;
        cout << endl;
        totalTris += bvhNodes[i].triCount;
    }

    cout << numTris << endl;
    cout << totalTris << endl;
#endif

    return rootNodeIdx;
}

void Scene::bvhUpdateNodeBounds(BvhNode& node)
{
    node.aabb = AABB();
    for (int i = 0; i < node.triCount; ++i)
    {
        const Triangle& leafTri = tris[bvhTriIdx[node.leftFirst + i]];
        node.aabb.grow(leafTri);
    }
}

float Scene::bvhEvaluateSAH(BvhNode& node, int axis, float pos)
{
    AABB leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for (int i = 0; i < node.triCount; ++i)
    {
        const Triangle& triangle = tris[bvhTriIdx[node.leftFirst + i]];
        if (triangle.centroid[axis] < pos)
        {
            ++leftCount;
            leftBox.grow(triangle);
        }
        else
        {
            ++rightCount;
            rightBox.grow(triangle);
        }
    }
    float cost = leftCount * leftBox.surfaceArea() + rightCount * rightBox.surfaceArea();
    return cost > 0 ? cost : FLT_MAX;
}

#define BVH_NUM_INTERVALS 8

struct Bin
{
    AABB aabb;
    int triCount = 0;
};

float Scene::bvhFindBestSplitPlane(BvhNode& node, int& axis, float& splitPos, AABB& leftChildBox, AABB& rightChildBox)
{
    float bestCost = FLT_MAX;
    for (int candidateAxis = 0; candidateAxis < 3; ++candidateAxis)
    {
        float axisMin = FLT_MAX;
        float axisMax = -FLT_MAX;
        for (int i = 0; i < node.triCount; ++i)
        {
            const Triangle& tri = tris[bvhTriIdx[node.leftFirst + i]];
            axisMin = min(axisMin, tri.centroid[candidateAxis]);
            axisMax = max(axisMax, tri.centroid[candidateAxis]);
        }

        if (axisMin == axisMax)
        {
            continue;
        }

        Bin bins[BVH_NUM_INTERVALS];
        float scale = BVH_NUM_INTERVALS / (axisMax - axisMin);
        for (int i = 0; i < node.triCount; ++i)
        {
            const Triangle& tri = tris[bvhTriIdx[node.leftFirst + i]];
            int binIdx = min(BVH_NUM_INTERVALS - 1, (int)((tri.centroid[candidateAxis] - axisMin) * scale));
            ++bins[binIdx].triCount;
            bins[binIdx].aabb.grow(tri);
        }

        AABB leftBoxes[BVH_NUM_INTERVALS - 1], rightBoxes[BVH_NUM_INTERVALS - 1];
        int leftCount[BVH_NUM_INTERVALS - 1], rightCount[BVH_NUM_INTERVALS - 1];
        AABB leftBox, rightBox;
        int leftSum = 0, rightSum = 0;
        for (int i = 0; i < BVH_NUM_INTERVALS - 1; ++i)
        {
            leftSum += bins[i].triCount;
            leftCount[i] = leftSum;
            leftBox.grow(bins[i].aabb);
            leftBoxes[i] = leftBox;
            rightSum += bins[BVH_NUM_INTERVALS - 1 - i].triCount;
            rightCount[BVH_NUM_INTERVALS - 2 - i] = rightSum;
            rightBox.grow(bins[BVH_NUM_INTERVALS - 1 - i].aabb);
            rightBoxes[BVH_NUM_INTERVALS - 2 - i] = rightBox;
        }

        scale = (axisMax - axisMin) / BVH_NUM_INTERVALS;
        for (int i = 0; i < BVH_NUM_INTERVALS - 1; ++i)
        {
            float cost = leftCount[i] * leftBoxes[i].surfaceArea() + rightCount[i] * rightBoxes[i].surfaceArea();
            if (cost < bestCost)
            {
                axis = candidateAxis;
                splitPos = axisMin + scale * (i + 1);
                leftChildBox = leftBoxes[i];
                rightChildBox = rightBoxes[i];
                bestCost = cost;
            }
        }
    }
    return bestCost;
}

void Scene::bvhSubdivide(BvhNode& node)
{
    int axis;
    float splitPos;
    AABB leftChildBox;
    AABB rightChildBox;
    float bestCost = bvhFindBestSplitPlane(node, axis, splitPos, leftChildBox, rightChildBox);

    if (bestCost >= node.cost())
    {
        return;
    }

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
    leftChild.aabb = leftChildBox;

    bvhNodes.emplace_back();
    BvhNode& rightChild = bvhNodes.back();
    rightChild.leftFirst = i;
    rightChild.triCount = node.triCount - leftCount;
    rightChild.aabb = rightChildBox;

    node.leftFirst = leftChildIdx;
    node.triCount = 0;

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
                newGeom.type = MESH;
                newGeom.bvhRootNodeIdx = loadMesh(fullPath);
            }
        }

        //link material
        Utils::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = Utils::tokenizeString(line);
            newGeom.materialId = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialId << "..." << endl;
        }

        glm::vec3 translation = glm::vec3(0);
        glm::vec3 rotation = glm::vec3(0);
        glm::vec3 scale = glm::vec3(1);
        int parentIdx = -1;

        //load transformations
        Utils::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = Utils::tokenizeString(line);

            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                if (tokens.size() == 2)
                {
                    scale = glm::vec3(atof(tokens[1].c_str()));
                }
                else
                {
                    scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
            } else if (strcmp(tokens[0].c_str(), "CHILD_OF") == 0) {
                parentIdx = atoi(tokens[1].c_str());
            }

            Utils::safeGetline(fp_in, line);
        }

        glm::mat4 thisGeomTransformMat = Utils::buildTransformationMatrix(translation, rotation, scale);
        if (parentIdx != -1)
        {
            thisGeomTransformMat = geoms[parentIdx].transform * thisGeomTransformMat;
        }
        newGeom.transform = thisGeomTransformMat;
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
        } else if (strcmp(tokens[0].c_str(), "FOCUS_DIST") == 0) {
            camera.focusDistance = atof(tokens[1].c_str());
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

int Scene::loadTexture(string filePath)
{
    if (textureIndices.find(filePath) != textureIndices.end())
    {
        return textureIndices[filePath];
    }

    cout << "Loading texture: " << filePath << endl;

    int textureIdx = textures.size();
    textures.emplace_back();
    Texture& texture = textures.back();

    int channels;
    unsigned char* textureData = stbi_load(filePath.c_str(), &texture.width, &texture.height, &channels, NULL);

    texture.host_dataPtr = new unsigned char [texture.width * texture.height * 4];

    // ensure all textures have 4 channels to simplify texture format logic
    if (channels < 4)
    {
        for (int i = 0; i < texture.width * texture.height; ++i)
        {
            for (int j = 0; j < channels; ++j)
            {
                texture.host_dataPtr[4 * i + j] = textureData[channels * i + j];
            }

            for (int j = channels; j < 4; ++j)
            {
                texture.host_dataPtr[4 * i + j] = 255;
            }
        }
    }
    else
    {
        memcpy(texture.host_dataPtr, textureData, texture.width * texture.height * 4);
    }

    texture.channels = 4;

    stbi_image_free(textureData);

    return textureIdx;
}

int Scene::loadMaterial(string materialId) {
    int id = atoi(materialId.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        string line;
        Utils::safeGetline(fp_in, line);
        while (!line.empty()) {
            vector<string> tokens = Utils::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "DIFF_COL") == 0) {
                if (tokens.size() == 2)
                {
                    newMaterial.diffuse.textureIdx = loadTexture(basePath + tokens[1]);
                }
                else
                {
                    newMaterial.diffuse.color = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
            } else if (strcmp(tokens[0].c_str(), "SPEC_EX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPEC_COL") == 0) {
                newMaterial.specular.color = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.specular.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.specular.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.specular.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMIT_COL") == 0) {
                newMaterial.emission.color = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "EMIT_STR") == 0) {
                newMaterial.emission.strength = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "NORM_MAP") == 0) {
                newMaterial.normalMap.textureIdx = loadTexture(basePath + tokens[1]);
            }

            Utils::safeGetline(fp_in, line);
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
