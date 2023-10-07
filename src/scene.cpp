#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "Mesh/objLoader.h"
#include <unordered_map>

string getDirectory(const string& path) {
    size_t found = path.find_last_of("/\\");
    return (found != string::npos) ? path.substr(0, found) : "";
}

string getBaseFilename(const string& path) {
    size_t start = path.find_last_of("/\\") + 1;
    size_t end = path.rfind(".");
    return path.substr(start, end - start);
}

Scene::Scene(string filename)
    : filename(filename), sqrtSamples(10)
{ 
    cout << "Reading scene from " << filename << "..." << endl;
    cout << " " << endl;
    
    // char* filename = (char*)filename.c_str();

    fp_in.open(filename);
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
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
    
    printf("Number of Geoms: %d\n", geoms.size());
    printf("\n");

#if USE_BVH
    LEAF_SIZE = static_cast<int>(log(geoms.size()));
    nBuckets = static_cast<int>(sqrt(geoms.size()));
    printf("Leaf size: %d\n", LEAF_SIZE);
    printf("Number of Buckets: %d\n", nBuckets);
    cout << "" << endl;
#endif

    // for stochastic sampling
    int sqrtSamples = static_cast<int>(sqrt(state.iterations));
    printf("Number of Samples: %d\n", sqrtSamples);
    cout << "" << endl;
}

Scene::~Scene()
{}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    cout << "Loading Geom " << id << "..." << endl;

    GeomType type = TRIANGLE;
    string objFilename = "";
    int materialid = -1;
    glm::vec3 translation, rotation, scale;
    glm::mat4 transform, inverseTransform, invTranspose;

    string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            type = SPHERE;
        }
        else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            type = CUBE;
        }
        else if (strcmp(line.c_str(), "obj") == 0) {
            cout << "Creating new obj..." << endl;
        }
        else {
            objFilename = line;
            cout << "Creating new obj..." << objFilename << endl;
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        materialid = atoi(tokens[1].c_str());
        cout << "Connecting Geom " << objectid << " to Material " << materialid << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        utilityCore::safeGetline(fp_in, line);
    }

    transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
    inverseTransform = glm::inverse(transform);
    invTranspose = glm::inverseTranspose(transform);

    if (type == CUBE || type == SPHERE) {
        geoms.push_back(Geom(type, materialid, translation, rotation, 
            scale, transform, inverseTransform, invTranspose));
    }
    else {
        string dir = getDirectory(filename);
        if (objFilename == "") {
            objFilename = getBaseFilename(filename);
        }
        string objFilePath = dir + "/" + objFilename + ".obj";

        loadObj(objFilePath, materialid, translation, rotation,
            scale, transform, inverseTransform, invTranspose);
    }
    
    return 1;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        } 
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
            camera.aperture = atof(tokens[1].c_str());
        } 
        else if (strcmp(tokens[0].c_str(), "FOCALDISTANCE") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
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
    int id = materials.size();
    cout << "Loading Material " << id << "..." << endl;
    Material newMaterial;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RGB") == 0) {
            glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            newMaterial.diffuse = color;
        }
        else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
            newMaterial.specular.exponent = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
            glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            newMaterial.specular.color = specColor;
        }
        else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
            newMaterial.hasReflective = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
            newMaterial.hasRefractive = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
            newMaterial.indexOfRefraction = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
            newMaterial.emittance = atof(tokens[1].c_str());
        }
    }
    materials.push_back(newMaterial);
    return 1;
}

int Scene::addObjMaterial(const tinyobj::material_t& mat) {
    cout << "Loading Obj Material: " << mat.name << "..." << endl;
    Material newMaterial;

    newMaterial.diffuse = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
    newMaterial.ambient = glm::vec3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
    newMaterial.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
    newMaterial.specular.exponent = mat.shininess;

    newMaterial.transmittance = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
    newMaterial.indexOfRefraction = mat.ior;
    newMaterial.roughness = mat.roughness; 

    newMaterial.metallic = mat.metallic;  
    newMaterial.sheen = mat.sheen; 
    // Decide on reflectivity
    float reflectivity = glm::length(newMaterial.specular.color);
    bool isMetallic = newMaterial.metallic > 0.5f;
    newMaterial.hasReflective = (reflectivity > 0.1f || isMetallic) ? 1.0f : 0.0f;

    // Decide on refractivity
    float opacity = 1.0f - glm::length(newMaterial.transmittance); // Opposite of transmittance
    newMaterial.hasRefractive = (opacity < 0.9f && newMaterial.indexOfRefraction != 1.0f) ? 1.0f : 0.0f;

    newMaterial.emittance = glm::length(glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]));

    int id = materials.size();
    materials.push_back(newMaterial);
    return id;
}


int Scene::loadObj(const string& objFilePath, int materialid,
    const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale,
    const glm::mat4& transform, const glm::mat4& inverseTransform,
    const glm::mat4& invTranspose) {

    ObjLoader objLoader;

    if (!objLoader.Load(objFilePath)) {
        cout << "Error Loading " << objFilePath << endl;
        return -1;
    }
    else {
        tinyobj::attrib_t attrib = objLoader.attrib;

        // Build an unordered map for materials based on their names
        std::unordered_map<std::string, tinyobj::material_t> materialMap;
        for (const auto& material : objLoader.materials) {
            materialMap[material.name] = material;
        }

        for (const auto& shape : objLoader.shapes) {
            // Check if shape's name is also a name of a material
            if (materialMap.find(shape.name) != materialMap.end()) {
                cout << "Loading Obj Shape: " << shape.name << endl;

                tinyobj::material_t material = materialMap[shape.name];
                materialid = addObjMaterial(material);

                cout << "Connection Obj Shape: " << shape.name << " to Material ID: " << materialid << endl;
                cout << "" << endl;

                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                    size_t fv = size_t(shape.mesh.num_face_vertices[f]);

                    // We need at least 3 vertices to form a triangle
                    if (fv < 3) {
                        index_offset += fv;
                        continue;
                    }

                    // Fetch the vertices of the polygon
                    std::vector<glm::vec3> vertices;
                    std::vector<glm::vec3> normals;
                    std::vector<glm::vec2> uvs;
                    for (size_t v = 0; v < fv; v++) {
                        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                        // Vertex positions
                        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                        vertices.push_back(glm::vec3(vx, vy, vz));

                        // Vertex normals
                        if (idx.normal_index != -1) {  // Check if the normal exists
                            tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                            tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                            tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                            normals.push_back(glm::vec3(nx, ny, nz));
                        }

                        // Vertex UVs (texture coordinates)
                        if (idx.texcoord_index != -1) {  // Check if the UV exists
                            tinyobj::real_t u = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                            tinyobj::real_t v = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                            uvs.push_back(glm::vec2(u, v));
                        }
                    }

                    // Triangulate the polygon using a fan triangulation
                    for (size_t v = 1; v < fv - 1; v++) {
                        Geom newGeom(TRIANGLE, materialid, translation, rotation,
                            scale, transform, inverseTransform, invTranspose);

                        newGeom.setVertices(vertices[0], vertices[v], vertices[v + 1]);
                        
                        if (normals.size() == vertices.size()) {
                            newGeom.setNormals(normals[0], normals[v], normals[v + 1]);
                        }

                        if (uvs.size() == vertices.size()) {
                            newGeom.setUVs(uvs[0], uvs[v], uvs[v + 1]);
                        }

                        geoms.push_back(newGeom);
                    }

                    index_offset += fv;
                }
            }
        }
    }
    return 1;
}



int Scene::partitionSplit(std::vector<BVHGeomInfo>& geomInfo, int start, int end, int dim, int geomCount,
    Bound& centroidBounds, Bound& bounds) {

    int mid = -1;

    if (geomCount <= 4) {
        // partition geoms into equally sized subset
        mid = start + (end - start) / 2;
        std::nth_element(&geomInfo[start], &geomInfo[mid], &geomInfo[end - 1] + 1,
            [dim](const BVHGeomInfo& a, const BVHGeomInfo& b) {
                return a.centroid[dim] < b.centroid[dim];
            });
    }
    else {
        // allocate buckets for SAH partition
        struct BucketInfo {
            int count = 0;
            Bound bounds;
        };
        std::vector<BucketInfo> buckets(nBuckets);

        // initialize bucket info
        for (int i = start; i < end; ++i) {
            int b = nBuckets * centroidBounds.offset(geomInfo[i].centroid)[dim];

            if (b == nBuckets) b = nBuckets - 1;
            buckets[b].count++;
            buckets[b].bounds = buckets[b].bounds.unionBound(geomInfo[i].bounds);
        }
        
        // compute cost for splitting after each bucket
        std::vector<float> cost(nBuckets - 1);
        float boundsSurfaceArea = bounds.computeBoxSurfaceArea();

        for (int i = 0; i < nBuckets - 1; ++i) {
            Bound b0, b1;
            int count0 = 0, count1 = 0;
            for (int j = 0; j < i; ++j) {
                b0 = b0.unionBound(buckets[j].bounds);
                count0 += buckets[j].count;
            }

            for (int j = i; j < nBuckets; ++j) {
                b1 = b1.unionBound(buckets[j].bounds);
                count1 += buckets[j].count;
            }

            cost[i] = 0.125f * (count0 * b0.computeBoxSurfaceArea() + 
                count1 * b1.computeBoxSurfaceArea()) / boundsSurfaceArea;
        }

        // find the bucket to split that minizes the cost
        float minSplitCost = cost[0];
        int minCostIdx = 0;

        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] < minSplitCost) {
                minSplitCost = cost[i];
                minCostIdx = i;
            }
        }

        // either create a leaf node or split geoms at selected SAH bucket
        float leafCost = static_cast<float>(geomCount);
        if (geomCount > LEAF_SIZE || minSplitCost < leafCost) {
            BVHGeomInfo* pmid = std::partition(&geomInfo[start], &geomInfo[end - 1] + 1,
                [=](const BVHGeomInfo& pi) {
                    int b = nBuckets * centroidBounds.offset(pi.centroid)[dim];
                    if (b == nBuckets) b = nBuckets - 1;

                    return b <= minSplitCost;
                });
            mid = pmid - &geomInfo[0];
        }
    }

    return mid;
}

BVHNode* Scene::constructBVHTree(std::vector<BVHGeomInfo>& geomInfo, int start, int end,
    int* totalNodes, std::vector<Geom>& orderedGeoms) {
    BVHNode* node = new BVHNode();
    (*totalNodes)++;

    // compute bounding box for all geoms from start to end in BVHNode
    Bound bounds;

    for (unsigned int i = start; i < end; ++i) {
        bounds = bounds.unionBound(geomInfo[i].bounds);
    }

    int geomCount = end - start;

    // leaf nodes
    if (geomCount <= LEAF_SIZE) {
        int geomIndex = orderedGeoms.size();
        for (int i = start; i < end; ++i) {
            int index = geomInfo[i].geomIndex;
            orderedGeoms.push_back(geoms[index]);
        }
        node->initLeaf(geomIndex, geomCount, bounds);
        return node;
    }
    else {
        // compute bounds for centroids, choose split dimension
        Bound centroidBounds;
        for (int i = start; i < end; ++i) {
            centroidBounds = centroidBounds.unionBound(geomInfo[i].centroid);
        }
        int dim = centroidBounds.getLongestAxis();

        if (centroidBounds.pMin[dim] == centroidBounds.pMax[dim]) {
            // create leaf node
            int geomIndex = orderedGeoms.size();
            for (int i = start; i < end; ++i) {
                int index = geomInfo[i].geomIndex;
                orderedGeoms.push_back(geoms[index]);
            }
            node->initLeaf(geomIndex, geomCount, bounds);
        }
        else {
            // partition geoms
            int mid = partitionSplit(geomInfo, start, end, dim, geomCount, centroidBounds, bounds);
            if (mid == -1 || mid == start) {
                mid = start + (end - start) / 2;
            }
            node->initInterior(dim, 
                constructBVHTree(geomInfo, start, mid, totalNodes, orderedGeoms),
                constructBVHTree(geomInfo, mid, end, totalNodes, orderedGeoms));
        }
    }

    return node;
}

int Scene::flattenBVHTree(BVHNode* node, int* offset) {
    LinearBVHNode* linearNode = &bvh[*offset];
    linearNode->bounds = node->bounds;

    int myOffset = (*offset)++;

    if (node->geomCount > 0) {
        // leaf node
        linearNode->geomIndex = node->geomIndex;
        linearNode->geomCount = node->geomCount;
    }
    else {
        linearNode->axis = node->splitAxis;
        linearNode->geomCount = 0;
        flattenBVHTree(node->left, offset);
        linearNode->rightChildOffset = flattenBVHTree(node->right, offset);
    }

    return myOffset;
}

void Scene::buildBVH() {
    std::vector<BVHGeomInfo> geomInfo;

    for (size_t i = 0; i < geoms.size(); ++i) {
        Bound bounds = geoms[i].getWorldBounds();
        geomInfo.push_back(BVHGeomInfo(i, bounds));
    } 
    
    int totalNodes = 0;
    std::vector<Geom> orderedGeoms;
    BVHNode* root = constructBVHTree(geomInfo, 0, geoms.size(), &totalNodes, orderedGeoms);

    printf("TotalNodes: %d\n", totalNodes);

    cout << "" << endl;

    geoms.swap(orderedGeoms);

    bvh.resize(totalNodes);
    int offset = 0;
    flattenBVHTree(root, &offset);

#if DEBUG_BVH
    // for debugging
    cout << "" << endl;
    printf("Dubugging for first root:\n");
    printf("Root\n");
    auto minBounds = root->bounds.pMin;
    auto maxBounds = root->bounds.pMax;

    printf("Min Bounds: (%f, %f, %f)\n", minBounds[0], minBounds[1], minBounds[2]);
    printf("Max Bounds: (%f, %f, %f)\n", maxBounds[0], maxBounds[1], maxBounds[2]);
    cout << "" << endl;

    auto node = bvh[0];
    minBounds = node.bounds.pMin;
    maxBounds = node.bounds.pMax;
    printf("Flatten\n");
    printf("Min Bounds: (%f, %f, %f)\n", minBounds[0], minBounds[1], minBounds[2]);
    printf("Max Bounds: (%f, %f, %f)\n", maxBounds[0], maxBounds[1], maxBounds[2]);
    cout << "" << endl;
#endif
}

