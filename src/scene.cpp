#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <stack>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <stb_image.h>

glm::vec3 multiplyMV(glm::mat4 m, glm::vec3 v) {
    return glm::vec3(m * glm::vec4(v, 1.0f));
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    lightMaterialNum = 0;
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
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0)
            {
                loadTexture(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECTMESH") == 0) {
                loadObj(tokens[1].c_str());
                cout << tokens[1] << endl;
            }
        }
    }

    if (triangles.size() > 0) {
        int level;
        root_node = buildBVH(0, triangles.size(), level, 0);

        reformatBVHToGPU();

        std::cout << "level: " << level << std::endl;
        std::cout << "num nodes: " << num_nodes << std::endl;
        std::cout << "Tri num: " << num_tris << std::endl;
    }
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
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                newGeom.textureId = atoi(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "BUMP") == 0) {
                const char* bumpPath = tokens[1].c_str();
                int bumpTextureId = loadTexture(bumpPath);
                if (bumpTextureId != -1) {
                    newGeom.bumpTextureId = bumpTextureId;
                }
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);



        if (newGeom.materialid < lightMaterialNum)
        {
            this->lightNum++;
            this->lights.push_back(newGeom);
        }

        geoms.push_back(newGeom);
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
        for (int i = 0; i < 9; i++) {
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
            else if (strcmp(tokens[0].c_str(), "SUBS") == 0) {
                newMaterial.hasSubsurface = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SUBSRADIUS") == 0)
            {
                newMaterial.subsurfaceRadius = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
                if (newMaterial.emittance > 0)
                {
                    ++lightMaterialNum;
                }
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

// Reference: https://github.com/tinyobjloader/tinyobjloader
int Scene::loadObj(const char* inputfile) {
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig reader_config;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    tinyobj::attrib_t attrib = reader.GetAttrib();
    std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    for (size_t i = 0; i < shapes.size(); i += 1) {
        printf("%s\n", shapes[i].name.c_str());
    }

    cout << inputfile << endl;
    Geom geo;
    geo.type = OBJMESH;
    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            geo.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            geo.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            geo.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
            geo.materialid = atoi(tokens[1].c_str());
        }

        utilityCore::safeGetline(fp_in, line);
    }

    geo.transform = utilityCore::buildTransformationMatrix(
        geo.translation, geo.rotation, geo.scale);
    geo.inverseTransform = glm::inverse(geo.transform);
    geo.invTranspose = glm::inverseTranspose(geo.transform);
   
    geo.triangleIdStart = triangles.size();

    float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;

    // Loop over shapes
    for (const auto& shape : shapes) {
        // Loop over faces
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            Triangle tri;
            tri.mat_ID = geo.materialid;
            glm::vec3 pos = glm::vec3(0.0f);
            glm::vec3 nor = glm::vec3(0.0f);
            glm::vec2 uv = glm::vec2(0.0f);

            for (size_t k = 0; k < 3; ++k) {

                if (shape.mesh.indices[i + k].vertex_index != -1) {
                    pos = glm::vec3(attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 0],
                        attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 1],
                        attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 2]);
                }

                if (shape.mesh.indices[i + k].texcoord_index != -1) {
                    uv = glm::vec2(
                        attrib.texcoords[2 * shape.mesh.indices[i + k].texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * shape.mesh.indices[i + k].texcoord_index + 1]
                    );
                }

                if (shape.mesh.indices[i + k].normal_index != -1) {
                    nor = glm::vec3(
                        attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 0],
                        attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 1],
                        attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 2]
                    );
                }

                tri.vertices[k] = pos;
                tri.normals[k] = nor;
                tri.uvs[k] = uv;
            }

            tri.vertices[0] = multiplyMV(geo.transform, tri.vertices[0]);
            tri.vertices[1] = multiplyMV(geo.transform, tri.vertices[1]);
            tri.vertices[2] = multiplyMV(geo.transform, tri.vertices[2]);
            tri.normals[0] = glm::normalize(multiplyMV(geo.invTranspose, tri.normals[0]));
            tri.normals[1] = glm::normalize(multiplyMV(geo.invTranspose, tri.normals[1]));
            tri.normals[2] = glm::normalize(multiplyMV(geo.invTranspose, tri.normals[2]));

            tri.plane_normal = glm::normalize(glm::cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[1]));
            tri.S = glm::length(glm::cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[1]));

            TriBounds newTriBounds;

            newTriBounds.tri_ID = num_tris;

            newTriBounds.AABB_max = glm::vec3(maxX, maxY, maxZ);
            newTriBounds.AABB_min = glm::vec3(minX, minY, minZ);

            float midX = (tri.vertices[0].x + tri.vertices[1].x + tri.vertices[2].x) / 3.0;
            float midY = (tri.vertices[0].y + tri.vertices[1].y + tri.vertices[2].y) / 3.0;
            float midZ = (tri.vertices[0].z + tri.vertices[1].z + tri.vertices[2].z) / 3.0;
            newTriBounds.AABB_centroid = glm::vec3(midX, midY, midZ);

            tri_bounds.push_back(newTriBounds);

            num_tris++;
            triangles.push_back(tri);
        }
    }

    geo.boundingBoxMax = glm::vec3(maxX, maxY, maxZ);
    geo.boundingBoxMin = glm::vec3(minX, minY, minZ);
    geo.triangleIdEnd = triangles.size();

    return 1;
}

glm::vec3 gridStripeNoise(float x, float y, float cellSize)
{
    int ix = static_cast<int>(x / cellSize);
    int iy = static_cast<int>(y / cellSize);

    if ((ix + iy) % 2 == 0)
    {
        return glm::vec3(1, 1, 1);
    }
    else
    {
        return glm::vec3(0, 0, 0);
    }
}

int Scene::loadTexture(string textureID)
{
    int id = atoi(textureID.c_str());
    std::cout << "Loading texture file: " << id << " starting index: " << textureColors.size() << endl;

    Texture texture;
    texture.id = id;
    texture.idx = textureColors.size();
    int width, height, channels;

    string line;
    utilityCore::safeGetline(fp_in, line);
    vector<string> tokens = utilityCore::tokenizeString(line);

    if (strcmp(tokens[0].c_str(), "PATH") == 0) {
        const char* filepath = tokens[1].c_str();
        unsigned char* img = stbi_load(filepath, &width, &height, &channels, 0);
        if (img != nullptr && width > 0 && height > 0)
        {
            texture.width = width;
            texture.height = height;
            texture.channel = channels;

            for (int i = 0; i < width * height; ++i)
            {
                glm::vec3 col = glm::vec3(img[3 * i + 0], img[3 * i + 1], img[3 * i + 2]) / 255.f;
                textureColors.emplace_back(col);
            }
        }
        stbi_image_free(img);
        textures.push_back(texture);
        return 1;
    }
    else if (strcmp(tokens[0].c_str(), "PROCEDURAL") == 0)
    {
        texture.width = 1000;
        texture.height = 1;
        texture.channel = 3;
        for (int j = 0; j < texture.height; j++)
        {
            for (int i = 0; i < texture.width; i++)
            {
                glm::vec3 col = gridStripeNoise(static_cast<float>(i), static_cast<float>(j), 10.f);
                textureColors.emplace_back(col);
            }
        }
        textures.push_back(texture);
        return 1;
    }
    std::cout << "Texture path does not exist" << endl;
    return -1;

}

BVHNode* Scene::buildBVH(int start_index, int end_index, int& level, int count) {
    BVHNode* new_node = new BVHNode();
    num_nodes++;
    int num_tris_in_node = end_index - start_index;

    glm::vec3 max_bounds(-FLT_MAX);
    glm::vec3 min_bounds(FLT_MAX);

    for (int i = start_index; i < end_index; ++i) {
        max_bounds = glm::max(max_bounds, tri_bounds[i].AABB_max);
        min_bounds = glm::min(min_bounds, tri_bounds[i].AABB_min);
    }

    // Dynamic termination: Depth
    const int MAX_DEPTH = 18;
    if (count > MAX_DEPTH) {
        for (int i = start_index; i < end_index; ++i) {
            mesh_tris_sorted.push_back(triangles[tri_bounds[i].tri_ID]);
        }
        new_node->tri_index = start_index;
        new_node->AABB_max = max_bounds;
        new_node->AABB_min = min_bounds;
        return new_node;
    }

    int mid_point = (start_index + end_index) / 2;
    //const int MIN_TRIANGLES = 4;
    //if (mid_point - start_index < MIN_TRIANGLES || end_index - mid_point < MIN_TRIANGLES) {
    //    for (int i = start_index; i < end_index; ++i) {
    //        mesh_tris_sorted.push_back(triangles[tri_bounds[i].tri_ID]);
    //    }
    //    new_node->tri_index = start_index;  // Assuming contiguous triangles
    //    new_node->AABB_max = max_bounds;
    //    new_node->AABB_min = min_bounds;
    //    return new_node;
    //}
    // leaf node
    /*if (num_tris_in_node <= 1) {
        mesh_tris_sorted.push_back(triangles[tri_bounds[start_index].tri_ID]);
        new_node->tri_index = mesh_tris_sorted.size() - 1;
        new_node->AABB_max = max_bounds;
        new_node->AABB_min = min_bounds;
        return new_node;
    }
    else {*/
    {
        glm::vec3 centroid_max(-FLT_MAX);
        glm::vec3 centroid_min(FLT_MAX);
        for (int i = start_index; i < end_index; ++i) {
            centroid_max = glm::max(centroid_max, tri_bounds[i].AABB_centroid);
            centroid_min = glm::min(centroid_min, tri_bounds[i].AABB_centroid);
        }
        glm::vec3 centroid_extent = centroid_max - centroid_min;

        int dimension_to_split = 0;
        if (centroid_extent.x >= centroid_extent.y && centroid_extent.x >= centroid_extent.z) {
            dimension_to_split = 0;
        }
        else if (centroid_extent.y >= centroid_extent.x && centroid_extent.y >= centroid_extent.z) {
            dimension_to_split = 1;
        }
        else {
            dimension_to_split = 2;
        }

        float centroid_midpoint = (centroid_min[dimension_to_split] + centroid_max[dimension_to_split]) / 2;

        if (centroid_min[dimension_to_split] == centroid_max[dimension_to_split]) {
            mesh_tris_sorted.push_back(triangles[tri_bounds[start_index].tri_ID]);
            new_node->tri_index = mesh_tris_sorted.size() - 1;
            new_node->AABB_max = max_bounds;
            new_node->AABB_min = min_bounds;
            return new_node;
        }

        TriBounds* pointer_to_partition_point = std::partition(&tri_bounds[start_index], &tri_bounds[end_index],
            [dimension_to_split, centroid_midpoint](const TriBounds& triangle_AABB) {
                return triangle_AABB.AABB_centroid[dimension_to_split] < centroid_midpoint;
            });

        mid_point = pointer_to_partition_point - &tri_bounds[0];

        if (level < count) { level = count; }

        new_node->child_nodes[0] = buildBVH(start_index, mid_point, level, count+1);
        new_node->child_nodes[1] = buildBVH(mid_point, end_index, level, count + 1);
        
        

        new_node->split_axis = dimension_to_split;
        new_node->tri_index = -1;

        new_node->AABB_max.x = glm::max(new_node->child_nodes[0]->AABB_max.x, new_node->child_nodes[1]->AABB_max.x);
        new_node->AABB_max.y = glm::max(new_node->child_nodes[0]->AABB_max.y, new_node->child_nodes[1]->AABB_max.y);
        new_node->AABB_max.z = glm::max(new_node->child_nodes[0]->AABB_max.z, new_node->child_nodes[1]->AABB_max.z);

        new_node->AABB_min.x = glm::min(new_node->child_nodes[0]->AABB_min.x, new_node->child_nodes[1]->AABB_min.x);
        new_node->AABB_min.y = glm::min(new_node->child_nodes[0]->AABB_min.y, new_node->child_nodes[1]->AABB_min.y);
        new_node->AABB_min.z = glm::min(new_node->child_nodes[0]->AABB_min.z, new_node->child_nodes[1]->AABB_min.z);
        return new_node;
    }
}


void Scene::reformatBVHToGPU() {
    BVHNode* cur_node;
    std::stack<BVHNode*> nodes_to_process;
    std::stack<int> index_to_parent;
    std::stack<bool> second_child_query;
    int cur_node_index = 0;
    int parent_index = 0;
    bool is_second_child = false;
    nodes_to_process.push(root_node);
    index_to_parent.push(-1);
    second_child_query.push(false);
    while (!nodes_to_process.empty()) {
        BVHNode_GPU new_gpu_node;

        cur_node = nodes_to_process.top();
        nodes_to_process.pop();
        parent_index = index_to_parent.top();
        index_to_parent.pop();
        is_second_child = second_child_query.top();
        second_child_query.pop();

        if (is_second_child && parent_index != -1) {
            bvh_nodes_gpu[parent_index].offset_to_second_child = bvh_nodes_gpu.size();
        }
        new_gpu_node.AABB_min = cur_node->AABB_min;
        new_gpu_node.AABB_max = cur_node->AABB_max;
        if (cur_node->tri_index != -1) {
            // leaf node
            new_gpu_node.tri_index = cur_node->tri_index;
        }
        else {
            // intermediate node
            new_gpu_node.axis = cur_node->split_axis;
            new_gpu_node.tri_index = -1;
            nodes_to_process.push(cur_node->child_nodes[1]);
            index_to_parent.push(bvh_nodes_gpu.size());
            second_child_query.push(true);
            nodes_to_process.push(cur_node->child_nodes[0]);
            index_to_parent.push(-1);
            second_child_query.push(false);
        }
        bvh_nodes_gpu.push_back(new_gpu_node);
    }
}