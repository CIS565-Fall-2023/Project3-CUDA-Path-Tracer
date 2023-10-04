#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <cmath>

#define EMISSIVE_FACTOR 1.5f

Scene::Scene(string filename) {
    //std::string file = filename.substr(filename.rfind('/') + 1);
    //fprintf(stderr, "filename is %s\n", filename.c_str());
    //fprintf(stderr, "file is %s\n", file.c_str());
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    if (fp_in.good() && !strcmp(strrchr(fname, '.'), ".gltf"))
    {
        load_gltf(filename);
    }
    else
    {
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
    }
}

bool Scene::gltf_load_materials(const tinygltf::Model &model)
{
    materials = std::vector<Material>();
    for (const auto& material : model.materials)
    {
        Material new_material;
        new_material.color = glm::vec3(static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0]),
                                       static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1]), 
                                       static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]));
        new_material.specular.exponent = 1.0f / material.pbrMetallicRoughness.roughnessFactor;
        new_material.specular.factor = material.pbrMetallicRoughness.metallicFactor;
        new_material.specular.color = new_material.color;
        new_material.hasReflective = (material.pbrMetallicRoughness.metallicFactor > 0.0f) ? 1.0f : 0.0f;
        new_material.hasRefractive = 0.0f;
        new_material.indexOfRefraction = 0.0f;
        new_material.emittance = (material.emissiveFactor.empty()) ? 0.0f : EMISSIVE_FACTOR; // hard coded emissive for now
        materials.push_back(new_material);
    }
    return true; // make it void?
}

// currently not using bvh here
bool Scene::load_gltf(string filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    string err;
    string warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
		fprintf(stderr, "warn1: %s \n", warn.c_str());
	}

    if (!err.empty()) {
        fprintf(stderr, "Err1: %s\n", err.c_str());
    }

    if (!ret) {
        fprintf(stderr, "Failed to parse glTF\n");
        return false;
    }

    //for (int i = 0; i < model.bufferViews.size(); i++)
    //{
    //    const tinygltf::BufferView& bufferView = model.bufferViews[i];
    //    // no support for sparse accessors at the moment
    //    const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
    //}
    if (model.scenes.size() <= 0)
    {
        fprintf(stderr, "No valid scenes\n");
        return false;
    }
    // for now, load one scene
    int display_scene = model.defaultScene > -1 ? model.defaultScene : 0;
    const tinygltf::Scene& scene = model.scenes[display_scene];
    std::array<std::string, 2> const supported_attributes = {
        "POSITION",
        "NORMAL"
        //TEXCOORD?
    };
    gltf_load_materials(model);
    //materials = std::vector<Material>();
    //for (const auto& material : model.materials)
    //{
    //    Material new_material;
    //    new_material.color = glm::vec3(static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0]),
    //        static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1]),
    //        static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]));
    //    new_material.specular.exponent = 1.0f / material.pbrMetallicRoughness.roughnessFactor;
    //    new_material.specular.factor = material.pbrMetallicRoughness.metallicFactor;
    //    new_material.specular.color = new_material.color;
    //    new_material.hasReflective = (material.pbrMetallicRoughness.metallicFactor > 0.0f) ? 1.0f : 0.0f;
    //    new_material.hasRefractive = 0.0f;
    //    new_material.indexOfRefraction = 0.0f;
    //    new_material.emittance = (material.emissiveFactor.empty()) ? 0.0f : EMISSIVE_FACTOR; // hard coded emissive for now
    //    materials.push_back(new_material);
    //}

    for (int i = 0; i < scene.nodes.size(); i++)
    {
        tinygltf::Node node = model.nodes[scene.nodes[i]];
        if (node.mesh < 0)
        {
            continue;
        }
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];


        // construct mesh and get transforms
        Geom new_mesh;
        new_mesh.type = MESH;
        glm::mat4 transformation_matrix = glm::mat4();
        glm::quat rotation_quat;

        if (node.matrix.size() == 16)
        {
            for (int i = 0; i < 16; i++) {
                transformation_matrix[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
            }
            glm::decompose(transformation_matrix, new_mesh.scale, rotation_quat, new_mesh.translation, glm::vec3(), glm::vec4());
        }
        else
        {
            if (!node.translation.empty())
            {
                new_mesh.translation = glm::vec3(static_cast<float>(node.translation[0]), 
                                                  static_cast<float>(node.translation[1]), 
                                                  static_cast<float>(node.translation[2]));
            }
            if (!node.rotation.empty())
            {
                rotation_quat = glm::quat(static_cast<float>(node.rotation[3]),
                                     static_cast<float>(node.rotation[0]),
                                     static_cast<float>(node.rotation[1]),
                                     static_cast<float>(node.rotation[2]));
            }
            if (!node.scale.empty())
            {
                new_mesh.scale = glm::vec3(static_cast<float>(node.scale[0]),
                                            static_cast<float>(node.scale[1]),
                                            static_cast<float>(node.scale[2]));
            }
		}
        new_mesh.rotation = glm::eulerAngles(rotation_quat);
        // TODO: link material
        for (const auto& primitive : mesh.primitives)
        {
            // initialize to nullptrs
            std::array<tinygltf::Accessor*, 2> accessors = {};
            std::array<tinygltf::BufferView*, 2> bufferViews = {};
            std::array<tinygltf::Buffer*, 2> buffers = {};
            std::array<float*, 2> data = {};

            for (int i = 0; i < supported_attributes.size(); i++)
            {
                if (primitive.attributes.find(supported_attributes[i]) != primitive.attributes.end())
                {
                    const int index = primitive.attributes.at(supported_attributes[i]);
                    accessors[i] = &model.accessors[index];
                    bufferViews[i] = &model.bufferViews[accessors[i]->bufferView];
                    buffers[i] = &model.buffers[bufferViews[i]->buffer];
                    data[i] = reinterpret_cast<float*>(&(buffers[i]->data[accessors[i]->byteOffset + bufferViews[i]->byteOffset]));
                }
            }
            // Assume each mesh is one material right now
            if (primitive.material >= 0)
            {
                new_mesh.materialid = static_cast<int>(primitive.material);
            }
            //if (primitive.attributes.find("POSITION") != primitive.attributes.end())
            //{
            //    const int pos_index = primitive.attributes.at("POSITION");
            //    const tinygltf::Accessor& pos_accessor = model.accessors[pos_index];
            //    const tinygltf::BufferView& pos_bufferView = model.bufferViews[accessor.bufferView];
            //    const tinygltf::Buffer& pos_buffer = model.buffers[bufferView.buffer];
            //    const float* pos_data = reinterpret_cast<const float*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
            //
            // not using indices
            if (primitive.indices < 0)
            {
                int pos_accessor_index = std::find(supported_attributes.begin(), supported_attributes.begin(), "POSITION") - supported_attributes.begin();
                for (int i = 0; i < (accessors[pos_accessor_index])->count; i += 3)
                {
                    Triangle tri;
                    for (int j = 0; j < supported_attributes.size(); j++)
                    {
                        if (accessors[j] == nullptr)
                        {
                            continue;
                        }
                        float* attribute_data = data[j];
                        if (supported_attributes[j] == "TEXCOORD_0")
                        {
                            // UV coords
                        }
                        else
                        {
                            glm::vec3 vec_0 = glm::vec3(attribute_data[i * 3], attribute_data[i * 3 + 1], attribute_data[i * 3 + 2]);
                            glm::vec3 vec_1 = glm::vec3(attribute_data[(i + 1) * 3], attribute_data[(i + 1) * 3 + 1], attribute_data[(i + 1) * 3 + 2]);
                            glm::vec3 vec_2 = glm::vec3(attribute_data[(i + 1) * 3], attribute_data[(i + 1) * 3 + 1], attribute_data[(i + 1) * 3 + 2]);

                            if (supported_attributes[j] == "POSITION")
                            {
                                tri.v0.pos = vec_0;
                                tri.v1.pos = vec_1;
                                tri.v2.pos = vec_2;
                                tri.centroid = (vec_0 + vec_1 + vec_2) * 0.333333333333f;

                            }
                            else if (supported_attributes[j] == "NORMAL")
                            {
                                tri.v0.nor = vec_0;
                                tri.v1.nor = vec_1;
                                tri.v2.nor = vec_2;
                            }
                        }
                    }
                    // For now assume that each mesh is one material
                    //tri.materialid = static_cast<int>(primitive.material);
                    tris.push_back(tri);
                }
            }
            // using indices
            else
            {
                const tinygltf::Accessor& index_accessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& index_bufferView = model.bufferViews[index_accessor.bufferView];
                const tinygltf::Buffer& index_buffer = model.buffers[index_bufferView.buffer];
                // assuming 16 bit indices - may need to change
                const uint16_t* index_data = reinterpret_cast<const uint16_t*>(&(index_buffer.data[index_accessor.byteOffset + index_bufferView.byteOffset]));
                for (int i = 0; i < index_accessor.count; i += 3)
                {

                    Triangle tri;
                    int index_0 = index_data[i];
                    int index_1 = index_data[i + 1];
                    int index_2 = index_data[i + 2];

                    for (int j = 0; j < supported_attributes.size(); j++)
                    {
                        if (accessors[j] == nullptr)
                        {
                            continue;
                        }
                        float* attribute_data = data[j];
                        if (supported_attributes[j] == "TEXCOORD_0")
                        {

                        }
                        else
                        {
                            glm::vec3 vec_0 = glm::vec3(attribute_data[index_0 * 3], attribute_data[index_0 * 3 + 1], attribute_data[index_0 * 3 + 2]);
                            glm::vec3 vec_1 = glm::vec3(attribute_data[index_1 * 3], attribute_data[index_1 * 3 + 1], attribute_data[index_1 * 3 + 2]);
                            glm::vec3 vec_2 = glm::vec3(attribute_data[index_2 * 3], attribute_data[index_2 * 3 + 1], attribute_data[index_2 * 3 + 2]);

                            if (supported_attributes[j] == "POSITION")
                            {
                                tri.v0.pos = vec_0;
                                tri.v1.pos = vec_1;
                                tri.v2.pos = vec_2;
                                tri.centroid = (vec_0 + vec_1 + vec_2) * 0.333333333333f;

                            }
                            else if (supported_attributes[j] == "NORMAL")
                            {
                                tri.v0.nor = vec_0;
                                tri.v1.nor = vec_1;
                                tri.v2.nor = vec_2;
                            }
                        }
                    }
                    tris.push_back(tri);
                }
            }
        }
        new_mesh.transform = utilityCore::buildTransformationMatrix(
            new_mesh.translation, new_mesh.rotation, new_mesh.scale);
        new_mesh.inverseTransform = glm::inverse(new_mesh.transform);
        new_mesh.invTranspose = glm::inverseTranspose(new_mesh.transform);
        geoms.push_back(new_mesh);
    }
    return true; // look at this
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

void Scene::bvh_build()
{
    bvh_nodes.reserve(tris.size() - 1);
    // assign triangle centroids at some other point, when loading in triangles perhaps
    tri_indices = std::vector<int>(tris.size());
    for (int i = 0; i < tri_indices.size(); i++)
    {
        tri_indices[i] = i;
    }
    bvh_nodes.emplace_back();
    root_node_index = bvh_nodes.size() - 1;
    nodes_used++;
    BvhNode& root = bvh_nodes[root_node_index];
    root.left_first = 0;
    root.tri_count = tri_indices.size();
    bvh_update_node_bounds(root_node_index);
    bvh_subdivide(root_node_index);
    bvh_reorder_tris();
    bvh_in_use = true;
}

void Scene::bvh_reorder_tris()
{
    std::vector<Triangle> temp_tris(tris);
 //   for (int i = 0; i < tri_indices.size(); i++)
 //   {
	//	tris[i] = temp_tris[tri_indices[i]];
	//}

    auto reorder = [&temp_tris](int index)
        {
            return temp_tris[index];
        };
    std::transform(tri_indices.begin(), tri_indices.end(), tris.begin(), reorder);
}

void Scene::bvh_update_node_bounds(uint32_t node_index)
{
    BvhNode& node = bvh_nodes[node_index];
    node.aa_bb.bmin = glm::vec3(FLT_MAX);
    node.aa_bb.bmax = glm::vec3(FLT_MIN);
    for (unsigned int first = node.left_first, i = 0; i < node.tri_count; i++)
    {
        Triangle& leaf_tri = tris[first + i];
        node.aa_bb.grow(leaf_tri);
    }
}

void Scene::bvh_subdivide(uint32_t node_index)
{
    BvhNode &node = bvh_nodes[node_index];
    // Determine split axis and position
    int axis;
    float split_pos;
    // Cost to split node
    float split_cost = bvh_find_best_split(node_index, axis, split_pos);
    // Cost of current node
    float no_split_cost = node.aa_bb.surface_area() * node.tri_count;
    // Only continue if split has a lower cost than not splitting
    if (split_cost > no_split_cost)
    {
		return;
	}   
    // Partition triangles contained in the node based on the proposed split
    int i = node.left_first;
    int j = i + node.tri_count - 1;
    while (i <= j)
    {
        if (tris[tri_indices[i]].centroid[axis] < split_pos)
        {
            i++;
        }
        else
        {
            std::swap(tri_indices[i], tri_indices[j--]);
        }
    }
    // Check if a side is empty, and if so, return
    int left_count = i - node.left_first;
    if (left_count == 0 || left_count == node.tri_count)
    {
		return;
	}
    // Create child nodes
    // Left
    bvh_nodes.emplace_back(node.left_first, left_count);
    int left_child_index = bvh_nodes.size() - 1;

    // Right
    bvh_nodes.emplace_back(i, node.tri_count - left_count);
    int right_child_index = bvh_nodes.size() - 1;

    // Update parent node & grow child node bounds
    node.left_first = left_child_index;
    node.tri_count = 0;
    bvh_update_node_bounds(left_child_index);
    bvh_update_node_bounds(right_child_index);

    // Recurse
    bvh_subdivide(left_child_index);
    bvh_subdivide(right_child_index);
}

// Returns split cost
float Scene::bvh_find_best_split(uint32_t node_index, int& axis, float& split_pos)
{
    float best_cost = FLT_MAX;
    BvhNode& node = bvh_nodes[node_index];
    for (int a = 0; a < 3; a++)
    {
        // Optimize bounding box to be defined by centroids
        float bounds_min = FLT_MAX;
        float bounds_max = FLT_MIN;
        for (unsigned int i = 0; i < node.tri_count; i++)
        {
            Triangle& tri = tris[tri_indices[node.left_first + i]];
            bounds_min = min(bounds_min, tri.centroid[a]);
            bounds_max = max(bounds_max, tri.centroid[a]);
        }
        if (bounds_min == bounds_max)
        {
			continue;
		}
        // Populate bins with triangles
        Bin bins[NUM_BINS];
        float scale = NUM_BINS / (bounds_max - bounds_min);
        for (unsigned int i = 0; i < node.tri_count; i++)
        {
            Triangle& tri = tris[tri_indices[node.left_first + i]];
			int bin_index = min(NUM_BINS - 1, (int)((tri.centroid[a] - bounds_min) * scale));
			bins[bin_index].tri_count++;
            bins[bin_index].bounds.grow(tri);
        }
        // Sweep over planes and gather data to calculate SAH
        float left_area[NUM_BINS - 1], right_area[NUM_BINS - 1];
        int left_count[NUM_BINS - 1], right_count[NUM_BINS - 1];
        Aabb left_box, right_box;
        int left_sum = 0, right_sum = 0;
        for (int i = 0; i < NUM_BINS - 1; i++)
        {
            left_sum += bins[i].tri_count;
            left_box.grow(bins[i].bounds);
            left_count[i] = left_sum;
            left_area[i] = left_box.area();

            right_sum += bins[NUM_BINS - 1 - i].tri_count;
            right_box.grow(bins[NUM_BINS - 1 - i].bounds);
            right_count[NUM_BINS - 2 - i] = right_sum;
            right_area[NUM_BINS - 2 - i] = right_box.area();
        }
        // Calculate SAH for each plane and evaluate which of the planes to return
        scale = (bounds_max - bounds_min) / NUM_BINS;
        for (int i = 0; i < NUM_BINS - 1; i++)
        {
            float plane_cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
            if (plane_cost < best_cost)
            {
                axis = a;
                split_pos = bounds_min + (i + 1) * scale;
                best_cost = plane_cost;
            }
        }
    }
    return best_cost;
}

bool Scene::using_bvh()
{
    return bvh_in_use;
}

