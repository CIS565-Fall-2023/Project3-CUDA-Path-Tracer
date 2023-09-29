#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "Mesh/objLoader.h"


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
    
    /*for (const auto& geom : geoms) {
        if (geom.type == TRIANGLE) {
            printf("V0: (%f, %f, %f)\n", geom.triangle.v0[0], geom.triangle.v0[1], geom.triangle.v0[2]);
            printf("V1: (%f, %f, %f)\n", geom.triangle.v1[0], geom.triangle.v1[1], geom.triangle.v1[2]);
            printf("V2: (%f, %f, %f)\n", geom.triangle.v2[0], geom.triangle.v2[1], geom.triangle.v2[2]);
            printf("\n");
        }
    }*/
    printf("Number of Geoms: %d\n", geoms.size());
    printf("\n");
}

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

                if (objLoader.Load("../scenes/cube.obj")) {
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

int Scene::getLongestAxis(const glm::vec3& minBounds, const glm::vec3& maxBounds) {
    glm::vec3 diff = maxBounds - minBounds;

    if (diff.x > diff.y && diff.x > diff.z) {
        return 0;
    }
    else if (diff.y > diff.z) {
        return 1;
    }
    else {
        return 2;
    }
}

float Scene::computeBoxSurfaceArea(const glm::vec3& min, const glm::vec3& max) {
    glm::vec3 diff = max - min;
    return 2.0f * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
}

int Scene::getBestSplit(std::vector<Geom> geoms, int start, int end) {
    // for prefix sums
    std::vector<glm::vec3> leftMins(end - start, glm::vec3(FLT_MAX));
    std::vector<glm::vec3> leftMaxs(end - start, glm::vec3(-FLT_MAX));
    std::vector<glm::vec3> rightMins(end - start, glm::vec3(FLT_MAX));
    std::vector<glm::vec3> rightMaxs(end - start, glm::vec3(-FLT_MAX));

    // Compute prefix sums for left splits
    for (int i = start; i < end - 1; i++) {
        glm::vec3 geomMin, geomMax;
        geoms[i].getBounds(geomMin, geomMax);
        if (i > start) {
            leftMins[i - start] = glm::min(leftMins[i - start - 1], geomMin);
            leftMaxs[i - start] = glm::max(leftMaxs[i - start - 1], geomMax);
        }
        else {
            leftMins[i - start] = geomMin;
            leftMaxs[i - start] = geomMax;
        }
    }

    // Compute prefix sums for right splits
    for (int i = end - 1; i > start; i--) {
        glm::vec3 geomMin, geomMax;
        geoms[i].getBounds(geomMin, geomMax);
        if (i < end - 1) {
            rightMins[i - start] = glm::min(rightMins[i - start + 1], geomMin);
            rightMaxs[i - start] = glm::max(rightMaxs[i - start + 1], geomMax);
        }
        else {
            rightMins[i - start] = geomMin;
            rightMaxs[i - start] = geomMax;
        }
    }

    float bestCost = FLT_MAX;
    int bestSplit = -1;

    for (int i = start; i < end - 1; i++) {
        float leftArea = computeBoxSurfaceArea(leftMins[i - start], leftMaxs[i - start]);
        float rightArea = computeBoxSurfaceArea(rightMins[i - start + 1], rightMaxs[i - start + 1]);
        float cost = leftArea * (i - start + 1) + rightArea * (end - i - 1);
        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = i;
        }
    }

    return bestSplit;

}

BVHNode* Scene::constructBVH(std::vector<Geom> geoms, int start, int end, 
    const int numLeaves) {
    std::unique_ptr<BVHNode> node = std::make_unique<BVHNode>();

    // compute bounding box for all geoms from start to end
    glm::vec3 minBounds(FLT_MAX);
    glm::vec3 maxBounds(FLT_MIN);  

    for (unsigned int i = start; i < end; ++i) {
        glm::vec3 geomMin, geomMax;
        geoms[i].getBounds(geomMin, geomMax);

        minBounds = glm::min(minBounds, geomMin);
        maxBounds = glm::max(maxBounds, geomMax);
    }

    node->minBounds = minBounds;
    node->maxBounds = maxBounds;

    // leaf nodes
    if (end - start <= numLeaves) { 
        // leaf node
        node->isLeafNode = true;
        node->geomIndex = start;
        node->numGeoms = end - start; 
        return node.release();
    }

    // choose the longest axis to split
    int axis = getLongestAxis(minBounds, maxBounds);

    // sort geoms by their centroids along the chosen axis
    std::sort(geoms.begin() + start, geoms.begin() + end, [axis](const Geom& a, const Geom& b) {
        glm::vec3 aMin, aMax, bMin, bMax;
        a.getBounds(aMin, aMax);
        b.getBounds(bMin, bMax);

        return 0.5f * (aMin[axis] + aMax[axis]) < 0.5f * (bMin[axis] + bMax[axis]);
    });

    // compute the best split with SAH cost
    int bestSplit = getBestSplit(geoms, start, end);
    if (bestSplit == -1) {
        // fall backs to midpoint split
        bestSplit = start + (end - start) / 2; 
    }

    // Recursively construct child nodes
    node->left = constructBVH(geoms, start, bestSplit + 1, numLeaves);
    node->right = constructBVH(geoms, bestSplit + 1, end, numLeaves);

    return node.release();
}

int Scene::flattenBVHTree(BVHNode* node) {
    if (!node) return 0;

    std::unique_ptr<CompactBVH> compactNode = std::make_unique<CompactBVH>();
    compactNode->minBounds = node->minBounds;
    compactNode->maxBounds = node->maxBounds;    
    compactNode->rightChildOffset = 0;

    bvh.push_back(*std::move(compactNode));
    int currentIndex = bvh.size() - 1;  // Store the index of the current node

    int leftSize = 0, rightSize = 0;

    if (node->isLeafNode) {
        bvh[currentIndex].geomStartIndex = node->geomIndex;
        bvh[currentIndex].geomEndIndex = node->geomIndex + node->numGeoms;
    }
    else {
        leftSize = flattenBVHTree(node->left);
        bvh[currentIndex].rightChildOffset = leftSize + 1;  // Update the correct node
        // printf("Stored node right child index: %d\n", bvh[currentIndex].rightChildOffset);
        rightSize = flattenBVHTree(node->right);
    }

    // Total size of the flattened subtree
    return 1 + leftSize + rightSize;
}

