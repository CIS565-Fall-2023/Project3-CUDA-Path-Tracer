#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "Mesh/objLoader.h"

const int LEAF_SIZE = 1;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << "..." << endl;
    cout << " " << endl;
    filename = (char*)filename.c_str();

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
}

Scene::~Scene()
{}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        std::vector<Geom> tempTriangles;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "obj") == 0) {
                cout << "Creating new obj..." << endl;

                ObjLoader objLoader;

                if (objLoader.Load("../scenes/teapot.obj")) {
                    loadObjGeom(objLoader.attrib, objLoader.shapes, tempTriangles);
                    printf("Triangle Size: %d\n", tempTriangles.size());
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

        for (auto& triangle : tempTriangles) {
            triangle.type = TRIANGLE;
            triangle.materialid = newGeom.materialid;
            triangle.transform = newGeom.transform;
            triangle.inverseTransform = newGeom.inverseTransform;
            triangle.invTranspose = newGeom.invTranspose;

            geoms.push_back(triangle);
        }

        if (newGeom.type == CUBE || newGeom.type == SPHERE) {
            geoms.push_back(newGeom);
        }

        return 1;
    }
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
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
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
}


void Scene::loadObjGeom(const tinyobj::attrib_t& attrib,
    const std::vector<tinyobj::shape_t>& shapes, std::vector<Geom>& tempTriangles) {

    // Loop over shapes
    for (const auto& shape : shapes) {
        cout << "Loading Obj Shape: " << shape.name << "\n" << endl;

        size_t index_offset = 0;
        // Loop over faces (polygons)
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);

            // We need at least 3 vertices to form a triangle
            if (fv < 3) {
                index_offset += fv;
                continue;
            }

            // Fetch the vertices of the polygon
            std::vector<glm::vec3> vertices;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                vertices.push_back(glm::vec3(vx, vy, vz));
            }

            // Triangulate the polygon using a fan triangulation
            for (size_t v = 1; v < fv - 1; v++) {
                Geom newGeom;
                newGeom.type = TRIANGLE;
                if (shape.mesh.material_ids.size() > 0) {
                    newGeom.materialid = shape.mesh.material_ids[f];
                }

                newGeom.triangle.v0 = vertices[0];
                newGeom.triangle.v1 = vertices[v];
                newGeom.triangle.v2 = vertices[v + 1];

                tempTriangles.push_back(newGeom);
            }

            index_offset += fv;
        }
    }
}


void Scene::loadObjMaterial(const std::vector<tinyobj::material_t>& tinyobjMaterials) {

    for (const auto& mat : tinyobjMaterials) {
        cout << "Loading Obj Material: " << mat.name << "...\n" << endl;
        Material newMaterial;

        auto diffuse = mat.diffuse;
        auto specular = mat.specular;
        auto emission = mat.emission;

        newMaterial.color = glm::vec3(diffuse[0], diffuse[1], diffuse[2]);
        newMaterial.specular.color = glm::vec3(specular[0], specular[1], specular[2]);
        newMaterial.specular.exponent = mat.shininess;
        newMaterial.indexOfRefraction = mat.ior;

        float reflectivity = glm::length(newMaterial.specular.color);
        newMaterial.hasReflective = (reflectivity > 0.1f) ? 1 : 0;
        newMaterial.hasRefractive = (mat.dissolve < 1.f) ? 1 : 0;

        newMaterial.emittance = glm::length(glm::vec3(emission[0], emission[1], emission[2]));

        materials.push_back(newMaterial);
    }

    cout << "Loaded Obj Material!" << endl;
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
        constexpr int nBuckets = 12;
        struct BucketInfo {
            int count = 0;
            Bound bounds;
        };
        BucketInfo buckets[nBuckets];

        // initialize bucket info
        for (int i = start; i < end; ++i) {
            int b = nBuckets * centroidBounds.offset(geomInfo[i].centroid)[dim];

            if (b == nBuckets) b = nBuckets - 1;
            buckets[b].count++;
            buckets[b].bounds = buckets[b].bounds.unionBound(geomInfo[i].bounds);
        }
        
        // compute cost for splitting after each bucket
        float cost[nBuckets - 1];
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

        for (int i = 1; i < nBuckets; ++i) {
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
        printf("Leaf Node Index: %d\n", geomIndex);
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
        Bound bounds = geoms[i].getBounds();
        geomInfo.push_back(BVHGeomInfo(i, bounds));
    } 

    printf("Geom Info Size: %d\n", geomInfo.size());
    
    int totalNodes = 0;
    std::vector<Geom> orderedGeoms;
    BVHNode* root = constructBVHTree(geomInfo, 0, geoms.size(), &totalNodes, orderedGeoms);

    printf("TotalNodes: %d\n", totalNodes);

    geoms.swap(orderedGeoms);

    bvh.resize(totalNodes);
    int offset = 0;
    flattenBVHTree(root, &offset);

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

}

