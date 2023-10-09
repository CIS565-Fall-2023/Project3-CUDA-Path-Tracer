#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE

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
            } else if (strcmp(tokens[0].c_str(), "OBJ_FILE") == 0) {
                loadObj(tokens[1].c_str());
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "GLTF_FILE") == 0) {
                loadGLTF(tokens[1].c_str());
                cout << " " << endl;
            }
        }
    }
    constructBVHforAllGeom();
}

void Scene::constructBVHforAllGeom()
{
    for (int i = 0; i < geoms.size(); i++) 
    {
        if (geoms[i].type == OBJ) 
        {
            buildBVH(geoms[i].triStart, geoms[i].triEnd);
        }
    }
}

int Scene::buildBVH(int start, int end)
{
    if (triangles.empty()) return -1;

    std::vector<BVHPrimitiveInfo> primInfo;

    for (int i = start; i < end; i++) 
    {
        const Triangle& tri = triangles[i];

        const glm::vec3& v1 = tri.verts[0];
        const glm::vec3& v2 = tri.verts[1];
        const glm::vec3& v3 = tri.verts[2];

        glm::vec3 centroid = (v1 + v2 + v3) * (1.0f / 3.0f);
        BVHPrimitiveInfo info(i, BBox(v1, v2, v3), centroid);
        primInfo.push_back(info);
    }

    int rootIdx = 0;
    int numNodes = (end - start) * 2 - 1;
    std::vector<BVHNode> nodes(numNodes);

    BVHNode& root = nodes[rootIdx];
    root.left = 0;
    root.right = 0;
    root.firstPrimOffset = 0;
    root.nPrimitives = end - start;

    updateBVHNode(primInfo, nodes, rootIdx);

    int nodesVisited = 1;
    int maxSize;
    int maxDepth = subdivide(primInfo, nodes, rootIdx, nodesVisited, maxSize);

    std::vector<Triangle> tempTris(end - start);
    for (int i = 0; i < end - start; i++) 
    {
        tempTris[i] = triangles[primInfo[i].id];
    }
    for (int i = 0; i < end - start; i++)
    {
        triangles[start + i] = tempTris[i];
    }

    bvhNodes.insert(bvhNodes.end(), nodes.begin(), nodes.begin() + nodesVisited);
    return maxDepth;
}

void Scene::updateBVHNode(const std::vector<BVHPrimitiveInfo>& primInfo, std::vector<BVHNode>& nodes, int idx)
{
    if (idx < 0 || idx >= nodes.size()) return;

    BVHNode& node = nodes[idx];
    node.bbox = BBox();
    
    for (int i = node.firstPrimOffset; i < node.firstPrimOffset + node.nPrimitives; i++) 
    {
        node.bbox = Union(node.bbox, primInfo[i].bbox);
    }
}

int Scene::subdivide(std::vector<BVHPrimitiveInfo>& primInfo, std::vector<BVHNode>& nodes, int idx, int& nodesVisited, int& maxSize)
{
    if (idx < 0 || idx >= nodes.size()) return -1;

    BVHNode& node = nodes[idx];
    node.splitAxis = node.bbox.MaximumExtent();

    int saxis = node.splitAxis;

    float splitPos = node.bbox.minP[saxis] + node.bbox.getDiagonal()[saxis] * 0.5f;

    int left = node.firstPrimOffset;
    int right = left + node.nPrimitives - 1;
    while (left <= right) 
    {
        if (primInfo[left].bbox.minP[saxis] < splitPos && primInfo[left].bbox.maxP[saxis] < splitPos)
            left++;
        else 
        {
            std::swap(primInfo[left], primInfo[right]);
            right--;
        }        
    }

    int leftNum = left - node.firstPrimOffset;
    if (leftNum == 0 || leftNum == node.nPrimitives) 
    {
        maxSize = node.nPrimitives;
        return 1;
    }

    int leftIdx = nodesVisited++;
    int rightIdx = nodesVisited++;
    node.left = leftIdx;
    node.right = rightIdx;

    nodes[leftIdx].firstPrimOffset = node.firstPrimOffset;
    nodes[leftIdx].nPrimitives = leftNum;
    nodes[rightIdx].firstPrimOffset = left;
    nodes[rightIdx].nPrimitives = node.nPrimitives - leftNum;
    node.nPrimitives = 0;

    updateBVHNode(primInfo, nodes, leftIdx);
    updateBVHNode(primInfo, nodes, rightIdx);

    int maxLeftSize;
    int maxRightSize;

    int leftDepth = subdivide(primInfo, nodes, leftIdx, nodesVisited, maxLeftSize);
    int rightDepth = subdivide(primInfo, nodes, rightIdx, nodesVisited, maxRightSize);

    maxSize = glm::max(maxLeftSize, maxRightSize);
    return glm::max(leftDepth, rightDepth) + 1;
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

        if (newGeom.materialid == 0) 
        {
            lights.push_back(newGeom);
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
        } else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOCAL_DIST") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
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

int Scene::loadObj(const char* filename)
{
    cout << "loading .obj file: " << filename << endl;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials; 
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

    if (!warn.empty()) cout << "WARN: " << warn << endl;
    if (!err.empty()) cout << "Err: " << err << endl;
    if (!ret) 
    {
        cout << "Failed to load .obj file. " << endl;
        return 0;
    }

    Geom newGeom;
    newGeom.type = OBJ;
    string line;

    // link material
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

    newGeom.triStart = triangles.size();

    BBox curBB;

    for (size_t i = 0; i < shapes.size(); i++) 
    {
        size_t index_offset = 0;

        Triangle tri;

        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
        {
            for (size_t v = 0; v < 3; v++)
            {
                tinyobj::index_t idx_t = shapes[i].mesh.indices[index_offset + v];
                size_t idx_v = (size_t)idx_t.vertex_index;
                tinyobj::real_t vx = attrib.vertices[3 * idx_v + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx_v + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx_v + 2];
                tri.verts[v] = glm::vec3(vx, vy, vz);

                if (idx_t.normal_index >= 0) 
                {
                    size_t idx_n = (size_t)idx_t.normal_index;
                    tinyobj::real_t nx = attrib.normals[3 * idx_n + 0];
                    tinyobj::real_t ny = attrib.normals[3 * idx_n + 1];
                    tinyobj::real_t nz = attrib.normals[3 * idx_n + 2];
                    tri.nors[v] = glm::vec3(nx, ny, nz);
                }

                if (idx_t.texcoord_index >= 0) 
                {
                    size_t idx_uv = (size_t)idx_t.texcoord_index;
                    tinyobj::real_t uvx = attrib.texcoords[2 * idx_uv + 0];
                    tinyobj::real_t uvy = attrib.texcoords[2 * idx_uv + 1];
                    tri.uvs[v] = glm::vec2(uvx, uvy);
                }          
            }
            index_offset += 3;
            tri.bbox = BBox(tri.verts[0], tri.verts[1], tri.verts[2]);
            triangles.push_back(tri);
            if (f == 0) curBB = tri.bbox;
            else curBB = Union(curBB, tri.bbox);
        }
    }

    newGeom.triEnd = triangles.size();
    newGeom.box = curBB;
    geoms.push_back(newGeom);

    return 1;
}

int Scene::loadGLTF(const char* filename)
{
    cout << "loading GLTF file: " << filename << endl;
    tinygltf::Model model;
    tinygltf::TinyGLTF reader;
    std::string warn;
    std::string err;

    bool ret = false;
    std::string ext1 = ".glb";
    std::string ext2 = ".gltf";
    std::string filestr = filename;
    if (equal(ext1.rbegin(), ext1.rend(), filestr.rbegin())) 
    {
        ret = reader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }
    else if (equal(ext2.rbegin(), ext2.rend(), filestr.rbegin())) 
    {
        ret = reader.LoadASCIIFromFile(&model, &err, &warn, filename);
    }

    if (!warn.empty()) cout << "WARN: " << warn << endl;
    if (!err.empty()) cout << "Err: " << err << endl;
    if (!ret)
    {
        cout << "Failed to load GLTF file. " << endl;
        return 0;
    }

    Geom newGeom;
    newGeom.type = OBJ;
    string line;

    // link material
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

    newGeom.triStart = triangles.size();

    BBox curBB;

    for (int m = 0; m < model.meshes.size(); m++) 
    {
        const tinygltf::Mesh& mesh = model.meshes[m];
        for (int p = 0; p < mesh.primitives.size(); p++) 
        {
            const tinygltf::Primitive& primitive = mesh.primitives[p];
            const tinygltf::Accessor& vertAcc = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& vertBufView = model.bufferViews[vertAcc.bufferView];
            const tinygltf::Buffer& vertBuf = model.buffers[vertBufView.buffer];
            const float* verts = reinterpret_cast<const float*>(&vertBuf.data[vertBufView.byteOffset + vertAcc.byteOffset]);

            const float* nors = nullptr; 
            if(primitive.attributes.find("NORMAL") != primitive.attributes.end())
            {
                const tinygltf::Accessor& norAcc = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& norBufView = model.bufferViews[norAcc.bufferView];
                const tinygltf::Buffer& norBuf = model.buffers[norBufView.buffer];
                nors = reinterpret_cast<const float*>(&norBuf.data[norBufView.byteOffset + norAcc.byteOffset]);
            }

            const float* uvs = nullptr;
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
            {
                const tinygltf::Accessor& uvAcc = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const tinygltf::BufferView& uvBufView = model.bufferViews[uvAcc.bufferView];
                const tinygltf::Buffer& uvBuf = model.buffers[uvBufView.buffer];
                uvs = reinterpret_cast<const float*>(&uvBuf.data[uvBufView.byteOffset + uvAcc.byteOffset]);
            }

            if (primitive.indices >= 0) 
            {
                const tinygltf::Accessor& idxAcc = model.accessors[primitive.indices];
                const tinygltf::BufferView& idxBufView = model.bufferViews[idxAcc.bufferView];
                const tinygltf::Buffer& idxBuf = model.buffers[idxBufView.buffer];
                const uint16_t* indices = reinterpret_cast<const uint16_t*>(&idxBuf.data[idxBufView.byteOffset + idxAcc.byteOffset]);

                Triangle tri;
                
                for (size_t i = 0; i < idxAcc.count; i += 3) 
                {
                    int idx0 = indices[i];
                    int idx1 = indices[i + 1];
                    int idx2 = indices[i + 2];

                    tri.verts[0] = glm::vec3(verts[idx0 * 3], verts[idx0 * 3 + 1], verts[idx0 * 3 + 2]);
                    tri.verts[1] = glm::vec3(verts[idx1 * 3], verts[idx1 * 3 + 1], verts[idx1 * 3 + 2]);
                    tri.verts[2] = glm::vec3(verts[idx2 * 3], verts[idx2 * 3 + 1], verts[idx2 * 3 + 2]);

                    if (nors) 
                    {
                        tri.nors[0] = glm::vec3(nors[idx0 * 3], nors[idx0 * 3 + 1], nors[idx0 * 3 + 2]);
                        tri.nors[1] = glm::vec3(nors[idx1 * 3], nors[idx1 * 3 + 1], nors[idx1 * 3 + 2]);
                        tri.nors[2] = glm::vec3(nors[idx2 * 3], nors[idx2 * 3 + 1], nors[idx2 * 3 + 2]);
                    }

                    if (uvs) 
                    {
                        tri.uvs[0] = glm::vec2(uvs[idx0 * 2], uvs[idx0 * 2 + 1]);
                        tri.uvs[1] = glm::vec2(uvs[idx1 * 2], uvs[idx1 * 2 + 1]);
                        tri.uvs[2] = glm::vec2(uvs[idx2 * 2], uvs[idx2 * 2 + 1]);
                    }

                    triangles.push_back(tri);
                    if (i == 0) curBB = BBox(tri.verts[0], tri.verts[1], tri.verts[2]);
                    else curBB = Union(curBB, BBox(tri.verts[0], tri.verts[1], tri.verts[2]));
                }
            }
            else 
            {
                Triangle tri;

                for (size_t i = 0; i < vertAcc.count; i += 3)
                {
                    int idx0 = i;
                    int idx1 = i + 1;
                    int idx2 = i + 2;

                    tri.verts[0] = glm::vec3(verts[idx0 * 3], verts[idx0 * 3 + 1], verts[idx0 * 3 + 2]);
                    tri.verts[1] = glm::vec3(verts[idx1 * 3], verts[idx1 * 3 + 1], verts[idx1 * 3 + 2]);
                    tri.verts[2] = glm::vec3(verts[idx2 * 3], verts[idx2 * 3 + 1], verts[idx2 * 3 + 2]);

                    if (nors)
                    {
                        tri.nors[0] = glm::vec3(nors[idx0 * 3], nors[idx0 * 3 + 1], nors[idx0 * 3 + 2]);
                        tri.nors[1] = glm::vec3(nors[idx1 * 3], nors[idx1 * 3 + 1], nors[idx1 * 3 + 2]);
                        tri.nors[2] = glm::vec3(nors[idx2 * 3], nors[idx2 * 3 + 1], nors[idx2 * 3 + 2]);
                    }

                    if (uvs)
                    {
                        tri.uvs[0] = glm::vec2(uvs[idx0 * 2], uvs[idx0 * 2 + 1]);
                        tri.uvs[1] = glm::vec2(uvs[idx1 * 2], uvs[idx1 * 2 + 1]);
                        tri.uvs[2] = glm::vec2(uvs[idx2 * 2], uvs[idx2 * 2 + 1]);
                    }

                    tri.bbox = BBox(tri.verts[0], tri.verts[1], tri.verts[2]);
                    triangles.push_back(tri);
                    if (i == 0) curBB = tri.bbox;
                    else curBB = Union(curBB, tri.bbox);
                }
            }
        }
    }

    newGeom.triEnd = triangles.size();
    newGeom.box = curBB;
    geoms.push_back(newGeom);

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
