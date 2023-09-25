#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>



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
                loadObject(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

namespace std {
    template <>
    struct hash<glm::vec3> {
        std::size_t operator()(const glm::vec3& vertex) const {
            return ((hash<float>()(vertex.x) ^
                (hash<float>()(vertex.y) << 1)) >> 1) ^
                (hash<float>()(vertex.z) << 1);
        }
    };
}

bool Scene::loadModel(const string& modelPath, int objectid)
{
    cout << "Loading Model " << modelPath << " ..." << endl;
    Object model;
    model.type = MODEL;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str());
    if (!warn.empty()) std::cout << warn << std::endl;

    if (!err.empty()) std::cerr << err << std::endl;

    if (!ret)  return false;

    std::unordered_map<glm::vec3, unsigned> vertex_set;
    for (const auto& shape : shapes)
    {
        for (const auto& index : shape.mesh.indices) 
        {
            glm::vec3 tmp;
            tmp.x = attrib.vertices[3 * index.vertex_index + 0];
            tmp.y = attrib.vertices[3 * index.vertex_index + 1];
            tmp.z = attrib.vertices[3 * index.vertex_index + 2];
            if (!vertex_set.count(tmp))
            {
                vertex_set[tmp] = verticies.size();
                verticies.emplace_back(tmp);
            }
        }
    }
    model.triangleStart = triangles.size();
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            assert(fv == 3);
            glm::ivec3 triangle;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                glm::vec3 tmp;
                tmp.x = attrib.vertices[3 * idx.vertex_index + 0];
                tmp.y = attrib.vertices[3 * idx.vertex_index + 1];
                tmp.z = attrib.vertices[3 * idx.vertex_index + 2];
                triangle[v] = vertex_set[tmp];
            }
            triangles.emplace_back(triangle);
            index_offset += fv;
        }
    }
    model.triangleEnd = triangles.size();

    std::string line;
    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        model.materialid = atoi(tokens[1].c_str());
        cout << "Connecting Geom " << objectid << " to Material " << model.materialid << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            model.Transform.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            model.Transform.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            model.Transform.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    model.Transform.transform = utilityCore::buildTransformationMatrix(
    model.Transform.translation, model.Transform.rotation, model.Transform.scale);
    model.Transform.inverseTransform = glm::inverse(model.Transform.transform);
    model.Transform.invTranspose = glm::inverseTranspose(model.Transform.transform);
    
    objects.emplace_back(model);

    return true;
}

bool Scene::loadGeometry(const string& type, int objectid)
{
    string line;
    Object newGeom;
    //load geometry type
    if (type == "sphere") {
        cout << "Creating new sphere..." << endl;
        newGeom.type = SPHERE;
    }
    else if (type == "cube") {
        cout << "Creating new cube..." << endl;
        newGeom.type = CUBE;
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
            newGeom.Transform.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.Transform.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.Transform.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.Transform.transform = utilityCore::buildTransformationMatrix(
        newGeom.Transform.translation, newGeom.Transform.rotation, newGeom.Transform.scale);
    newGeom.Transform.inverseTransform = glm::inverse(newGeom.Transform.transform);
    newGeom.Transform.invTranspose = glm::inverseTranspose(newGeom.Transform.transform);

    objects.push_back(newGeom);
    return true;
}

int Scene::loadObject(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != objects.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Object " << id << "..." << endl;
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "geometry")//load geometry
            {
                loadGeometry(tokens[1], id);
            }
            else//load model
            {
                loadModel(tokens[1], id);
            }
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
        for (int i = 0; i < 3; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                float ior = atof(tokens[1].c_str());
                if (ior != 0.0) newMaterial.type |= MaterialType::frenselSpecular;
                newMaterial.indexOfRefraction = ior;
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                float emittance = atof(tokens[1].c_str());
                if (emittance != 0.0) newMaterial.type |= MaterialType::emitting;
                newMaterial.emittance = emittance;
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

void Scene::buildBVH()
{
    for (int i=0;i<objects.size();i++)
    {
        const Object& obj = objects[i];
        if (obj.type == MODEL)
        {
            for (int j = obj.triangleStart; j != obj.triangleEnd; j++)
            {
                primitives.emplace_back(obj, i, j, &triangles[0], &verticies[0]);
            }
        }
        else
        {
            primitives.emplace_back(obj, i);
        }
    }
    bvhroot = buildBVHTreeRecursiveSAH(primitives, 0, primitives.size());
    assert(checkBVHTreeFull(bvhroot));
}

void Scene::buildStacklessBVH()
{
    compactBVHTreeForStacklessTraverse(bvhArray, bvhroot);
}
