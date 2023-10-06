#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
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
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

    if (hasMesh)
    {
        cout << "Building BVH..." << endl;
        BuildBVH();
        cout << "BVH nodes: " << bvhNode.size() << endl;
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
        string meshPath;
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
            else if (strcmp(line.c_str(), "mesh") == 0)
            {
                newGeom.type = MESH;         
                utilityCore::safeGetline(fp_in, line);
                vector<string> tokens = utilityCore::tokenizeString(line);
                meshPath = tokens[0];
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

        if (newGeom.type == MESH)
        {        
            cout << "Creating new mesh..." << endl;
            hasMesh = true;
            if (loadOBJ(meshPath, newGeom) != 1) {
                cout << "loadOBJ Failed." << endl;
            }
        }

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

glm::vec3 multiplyMV2(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

int Scene::loadOBJ(string filePath, Geom& mesh) {
    tinyobj::ObjReader reader;
    reader.ParseFromFile(filePath);
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    AABB aabb;
    aabb.max = glm::vec3(FLT_MIN);
    aabb.min = glm::vec3(FLT_MAX);

    mesh.triIdx = tri.size();
    int triCnt = 0;
    for (int i = 0; i < shapes.size(); i++)
    {      
        int faceSize = shapes[i].mesh.material_ids.size();
        auto& indices = shapes[i].mesh.indices;
        for (int j = 0; j < faceSize; j++)
        {
            Triangle t;
            for (int k = 0; k < 3; k++)
            {
                int idx = indices[3 * j + k].vertex_index;
                t.v[k] = glm::vec3(attrib.vertices[3 * idx + 0], attrib.vertices[3 * idx + 1], attrib.vertices[3 * idx + 2]);

                t.v[k] = multiplyMV2(mesh.transform, glm::vec4(t.v[k], 1.0f));

                aabb.min = glm::min(aabb.min, t.v[k]);
                aabb.max = glm::max(aabb.max, t.v[k]);

                if (attrib.normals.size() > 0)
                {
                    int idx_n = indices[3 * j + k].normal_index;
                    t.n[k] = glm::vec3(attrib.normals[3 * idx_n + 0], attrib.normals[3 * idx_n + 1], attrib.normals[3 * idx_n + 2]);      
                    t.n[k] = multiplyMV2(mesh.invTranspose, glm::vec4(t.v[k], 0.0f));
                }
                if (attrib.texcoords.size() > 0)
                {
                    int idx_t = indices[3 * j + k].texcoord_index;
                    t.uv[k] = glm::vec2(attrib.texcoords[2 * idx_t + 0], attrib.texcoords[2 * idx_t + 1]);
                }
            }
            t.geomIdx = geoms.size();
            tri.push_back(t);
            triCnt++;
        }
    }
    mesh.aabb = aabb;
    mesh.triCnt = triCnt;
    cout << "Loaded OBJ!"<< endl;
    return 1;
}

void Scene::BuildBVH() {
    N = tri.size();
    cout << "N ="<< N << endl;
    for (int i = 0; i < N; i++) triIdx.push_back(i);

    for (int i = 0; i < N; i++) {
        tri[i].centroid = (tri[i].v[0] + tri[i].v[1] + tri[i].v[2]) * 0.3333f;
    }
/*
    for (int i = 0; i < N * 2 - 1; i++) {
        BVHNode node;
        bvhNode.push_back(node);
    }
*/
    // assign all triangles to root node
    BVHNode root;
    root.leftNode = 0;
    root.firstTriIdx = 0;
    root.triCount = N;
    bvhNode.push_back(root);

    UpdateNodeBounds(rootNodeIdx);
    Subdivide(rootNodeIdx);     
}

void Scene::UpdateNodeBounds(int nodeIdx) {
    BVHNode& node = bvhNode[nodeIdx];
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);
    int first = node.firstTriIdx;
    for (int i = 0; i < node.triCount; i++)
    {
        int leafTriIdx = triIdx[first + i];
        Triangle leafTri = tri[leafTriIdx];
        node.aabbMin = glm::min(node.aabbMin, leafTri.v[0]);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v[1]);
        node.aabbMin = glm::min(node.aabbMin, leafTri.v[2]);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v[0]);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v[1]);
        node.aabbMax = glm::max(node.aabbMax, leafTri.v[2]);
    }
}

void Scene::Subdivide(int nodeIdx) {
    // terminate recursion
    BVHNode& node = bvhNode[nodeIdx];
    if (node.triCount <= 2) {       
        return;
    }

    // determine split axis and position
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
    // in-place partition
    int i = node.firstTriIdx;
    int j = i + node.triCount - 1;
    while (i <= j)
    {
        if (tri[triIdx[i]].centroid[axis] < splitPos)
            i++;
        else
            swap(triIdx[i], triIdx[j--]);
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.firstTriIdx;
    if (leftCount == 0 || leftCount == node.triCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    BVHNode left;
    left.firstTriIdx = node.firstTriIdx;
    left.triCount = leftCount;
    bvhNode.push_back(left);

    BVHNode right;
    right.firstTriIdx = i;
    right.triCount = node.triCount - leftCount;
    bvhNode.push_back(right);

    node.leftNode = leftChildIdx;
    node.triCount = 0;
    UpdateNodeBounds(leftChildIdx);
    UpdateNodeBounds(rightChildIdx);
    // recurse
    Subdivide(leftChildIdx);
    Subdivide(rightChildIdx);
}


