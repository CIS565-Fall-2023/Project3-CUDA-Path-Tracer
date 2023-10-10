#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION

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
            }
        }
    }

#if BVH
    //build bvh tree for entire scene
    cout << "Constructing BVH tree for scene" << endl;
    buildBvhTree();
    cout << "Finished parsing scene" << endl;
#endif
}
#if BVH
void Scene::print_tree(BVHNode& node) {
    if (node.leftNode != -1) {
        cout << " ------ left node ------ " << endl;
        print_tree(bvh_nodes[node.leftNode]);
    }
    cout << "--- node ---" << endl;
    cout << glm::to_string(node.min) << ", " << glm::to_string(node.max) << endl;
    cout << node.leftNode << endl;
    cout << node.triIndexStart << ", " << node.triCount << endl;
    cout << node.geomIndexStart << ", " << node.geomCount << endl;
    if (node.leftNode != -1) {
        cout << " ------ right node ------ " << endl;
        print_tree(bvh_nodes[node.leftNode+1]);
    }
}

void Scene::buildBvhTree() {
    //global root bbs
    initBvhIndexArrs();
    //calculate min and max for root(same code as the grow but im lazy to refactor)
    glm::vec3 g_bb_min, g_bb_max;
    for (int i = 0; i < bvh_tri_indices.size(); i++) {
        BVHTriIndex bti = bvh_tri_indices[i];
        Triangle& tri = mesh_triangles[bti.triIndex];
        Geom& g = geoms[tri.mesh_index];
        glm::vec3 p1 = tri.points[0].pos;
        glm::vec3 p2 = tri.points[1].pos;
        glm::vec3 p3 = tri.points[2].pos;

        g_bb_min = glm::min(g_bb_min, p1);
        g_bb_max = glm::max(g_bb_max, p1);
        g_bb_min = glm::min(g_bb_min, p2);
        g_bb_max = glm::max(g_bb_max, p2);
        g_bb_min = glm::min(g_bb_min, p3);
        g_bb_max = glm::max(g_bb_max, p3);
    }

    //for geoms as well
    for (int i = 0; i < bvh_geom_indices.size(); i++) {
        BVHGeomIndex bgi = bvh_geom_indices[i];
        Geom& g = geoms[bgi.geomIndex];
        //to grow bb for geoms, need to grow on the min and max corners for spheres + cubes
        // since scaling on individual axes is a thing need to find max and min of transformed standard bbs
        // buh there has to be an easier way to do this
        glm::vec3 vs[8];
        vs[0] = glm::vec3(-0.5, -0.5, -0.5);
        vs[1] = glm::vec3(-0.5, -0.5, 0.5);
        vs[2] = glm::vec3(-0.5, 0.5, -0.5);
        vs[3] = glm::vec3(-0.5, 0.5, 0.5);
        vs[4] = glm::vec3(0.5, -0.5, -0.5);
        vs[5] = glm::vec3(0.5, -0.5, 0.5);
        vs[6] = glm::vec3(0.5, 0.5, -0.5);
        vs[7] = glm::vec3(0.5, 0.5, 0.5);
        for (int j = 0; j < 8; j++) {
            glm::vec3 transformed = glm::vec3(g.transform * glm::vec4(vs[j], 1.f));
            g_bb_min = glm::min(g_bb_min, transformed);
            g_bb_max = glm::max(g_bb_max, transformed);
        }
    }
    //root node
    BVHNode root_node = { g_bb_min, g_bb_max, -1, 0, bvh_tri_indices.size(), 0, bvh_geom_indices.size() };
    bvh_nodes.push_back(root_node);
    subdivide_bvh(bvh_nodes[0]);
}

void Scene::initBvhIndexArrs() {
    //create array for bvh operations
    for (int i = 0; i < mesh_triangles.size(); i++) {
        Triangle& tri = mesh_triangles[i];
        BVHTriIndex bti;
        bti.triIndex = i;
        glm::mat4 local_to_global = geoms[tri.mesh_index].transform;
        //calculate centroid by transforming local frame pts to global, then average
        bti.gFrameCentroid = (glm::vec3(local_to_global * glm::vec4(tri.points[0].pos, 1.f)) + glm::vec3(local_to_global * glm::vec4(tri.points[1].pos, 1.f)) + glm::vec3(local_to_global * glm::vec4(tri.points[2].pos, 1.f))) / 3.f;
        bvh_tri_indices.push_back(bti);
    }
    //do the same for non mesh geoms
    for (int i = 0; i < geoms.size(); i++) {
        Geom& g = geoms[i];
        if (g.type != MESH_PRIM) {
            BVHGeomIndex bgi;
            bgi.geomIndex = i;
            //spheres, cubes centered at origin pre transform
            bgi.gFrameCentroid = glm::vec3(g.transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
            bvh_geom_indices.push_back(bgi);
        }
    }
}

float Scene::eval_sah(BVHNode& node, int axis, float pos, glm::vec3 &l_bb_min, glm::vec3& l_bb_max, 
    glm::vec3& r_bb_min, glm::vec3& r_bb_max) {
    int left_count = 0, right_count = 0;
    // count and grow bb for left and right divisions from tris
    for (int i = 0; i < node.triCount; i++) {
        BVHTriIndex bti = bvh_tri_indices[node.triIndexStart + i];
        Triangle& tri = mesh_triangles[bti.triIndex];
        Geom& g = geoms[tri.mesh_index];
        // need to be in global frame for bbs
        glm::vec3 p1 = tri.points[0].pos;
        glm::vec3 p2 = tri.points[1].pos;
        glm::vec3 p3 = tri.points[2].pos;
        if (bti.gFrameCentroid[axis] < pos) {
            left_count++;
            l_bb_min = glm::min(l_bb_min, p1);
            l_bb_max = glm::max(l_bb_max, p1);
            l_bb_min = glm::min(l_bb_min, p2);
            l_bb_max = glm::max(l_bb_max, p2);
            l_bb_min = glm::min(l_bb_min, p3);
            l_bb_max = glm::max(l_bb_max, p3);
        }
        else {
            right_count++;
            r_bb_min = glm::min(r_bb_min, p1);
            r_bb_max = glm::max(r_bb_max, p1);
            r_bb_min = glm::min(r_bb_min, p2);
            r_bb_max = glm::max(r_bb_max, p2);
            r_bb_min = glm::min(r_bb_min, p3);
            r_bb_max = glm::max(r_bb_max, p3);
        }
    }
    // continue for geoms
    for (int i = 0; i < node.geomCount; i++) {
        BVHGeomIndex bgi = bvh_geom_indices[node.geomIndexStart + i];
        Geom& g = geoms[bgi.geomIndex];
        //to grow bb for geoms, need to grow on the min and max corners for spheres + cubes
        // since scaling on individual axes is a thing need to find max and min of transformed standard bbs
        // buh there has to be an easier way to do this
        glm::vec3 vs[8];
        vs[0] = glm::vec3(-0.5, -0.5, -0.5);
        vs[1] = glm::vec3(-0.5, -0.5, 0.5);
        vs[2] = glm::vec3(-0.5, 0.5, -0.5);
        vs[3] = glm::vec3(-0.5, 0.5, 0.5);
        vs[4] = glm::vec3(0.5, -0.5, -0.5);
        vs[5] = glm::vec3(0.5, -0.5, 0.5);
        vs[6] = glm::vec3(0.5, 0.5, -0.5);
        vs[7] = glm::vec3(0.5, 0.5, 0.5);
        for (int j = 0; j < 8; j++) {
            glm::vec3 transformed = glm::vec3(g.transform * glm::vec4(vs[j], 1.f));
            if (bgi.gFrameCentroid[axis] < pos) {
                left_count++;
                l_bb_min = glm::min(l_bb_min, transformed);
                l_bb_max = glm::max(l_bb_max, transformed);
            }
            else {
                right_count++;
                r_bb_min = glm::min(r_bb_min, transformed);
                r_bb_max = glm::max(r_bb_max, transformed);
            }
        }
    }

    //calculate cost for split
    glm::vec3 r_span = r_bb_max - r_bb_min;
    glm::vec3 l_span = l_bb_max - l_bb_min;

    float cost = left_count * (l_span.x * l_span.y + l_span.y * l_span.z + l_span.z * l_span.x) + right_count * (r_span.x * r_span.y + r_span.y * r_span.z + r_span.z * r_span.x);
    return cost > 0 ? cost : FLT_MAX;
}

void Scene::subdivide_bvh(BVHNode& node) {
    // determine best place to split thru sah
    int best_axis = -1;
    float best_split = 0, best_cost = FLT_MAX;
    glm::vec3 best_lbb_min, best_lbb_max, best_rbb_min, best_rbb_max;
    for (int axis = 0; axis < 3; axis++) {
        //test all tri centroids for best sah
        for (int i = 0; i < node.triCount; i++) {
            glm::vec3 centroid = bvh_tri_indices[node.triIndexStart + i].gFrameCentroid;
            glm::vec3 l_bb_min(FLT_MAX), l_bb_max(FLT_MIN), r_bb_min(FLT_MAX), r_bb_max(FLT_MIN);
            float cost = eval_sah(node, axis, centroid[axis], l_bb_min, l_bb_max, r_bb_min, r_bb_max);
            if (cost < best_cost) {
                best_cost = cost;
                best_split = centroid[axis];
                best_axis = axis;
                best_lbb_min = l_bb_min;
                best_lbb_max = l_bb_max;
                best_rbb_min = r_bb_min;
                best_rbb_max = r_bb_max;
            }
        }

        //test all geom centroids for best sah
        for (int i = 0; i < node.geomCount; i++) {
            glm::vec3 centroid = bvh_geom_indices[node.geomIndexStart + i].gFrameCentroid;
            glm::vec3 l_bb_min(FLT_MAX), l_bb_max(FLT_MIN), r_bb_min(FLT_MAX), r_bb_max(FLT_MIN);
            float cost = eval_sah(node, axis, centroid[axis], l_bb_min, l_bb_max, r_bb_min, r_bb_max);
            if (cost < best_cost) {
                best_cost = cost;
                best_split = centroid[axis];
                best_axis = axis;
                best_lbb_min = l_bb_min;
                best_lbb_max = l_bb_max;
                best_rbb_min = r_bb_min;
                best_rbb_max = r_bb_max;
            }
        }
    }

    // if best sah is worse than cur parent one dont split and return
    glm::vec3 parent_dim = node.max - node.min;
    float parent_area = parent_dim.x * parent_dim.y + parent_dim.y * parent_dim.z + parent_dim.z * parent_dim.x;
    float parent_sah_cost = (node.triCount + node.geomCount) * parent_area;
    if (best_cost >= parent_sah_cost) return;

    //split on the best axis and pos
    //partition tri and geom indices
    int i = node.triIndexStart;
    int j = i + node.triCount - 1;
    while (i <= j) {
        if (bvh_tri_indices[i].gFrameCentroid[best_axis] < best_split) {
            i++;
        }
        else {
            std::swap(bvh_tri_indices[i], bvh_tri_indices[j--]);
        }
    }
    int rchild_tri_start = i;
    i = node.geomIndexStart;
    j = i + node.geomCount - 1;
    while (i <= j) {
        if (bvh_geom_indices[i].gFrameCentroid[best_axis] < best_split) {
            i++;
        }
        else {
            std::swap(bvh_geom_indices[i], bvh_geom_indices[j--]);
        }
    }
    int rchild_geom_start = i;

    //create new child nodes + link them to parent
    BVHNode l_node = { best_lbb_min, best_lbb_max, -1, node.triIndexStart, rchild_tri_start - node.triIndexStart, 
        node.geomIndexStart, rchild_geom_start - node.geomIndexStart };
    BVHNode r_node = { best_rbb_min, best_rbb_max, -1, rchild_tri_start, node.triCount - l_node.triCount,
        rchild_geom_start, node.geomCount - l_node.geomCount };
    int lnindex = bvh_nodes.size();
    node.leftNode = lnindex; //right node will always be at this index +1
    bvh_nodes.push_back(l_node);
    bvh_nodes.push_back(r_node);

    subdivide_bvh(bvh_nodes[lnindex]);
    subdivide_bvh(bvh_nodes[lnindex +1]);
}

#endif
// from tinygltf example basic
// recursively iter through scene nodes to parse meshes and associated maps
void Scene::parseMesh(tinygltf::Model& model, tinygltf::Mesh& mesh, std::vector<Geom>& newGeoms, glm::mat4 tmat) {
    //1 Geom per prim(submesh)
    for (const tinygltf::Primitive& prim : mesh.primitives) {
        //standardized data structs
        Geom mesh_geom;
        mesh_geom.type = MESH_PRIM;
        std::vector<Triangle> curr_tris;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<unsigned int> curr_indices;
        std::vector<glm::vec2> uvs;
        std::vector<MeshPoint> curr_mesh_pts;
        for (const std::pair<const std::string, int>& attribute : prim.attributes) {
            //locate corresponding accessor, buf view, buf
            const tinygltf::Accessor& accessor = model.accessors[attribute.second];
            const tinygltf::BufferView& buf_view = model.bufferViews[accessor.bufferView];
            tinygltf::Buffer& buf = model.buffers[buf_view.buffer];

            // offset to where data begins for this attribute
            unsigned char* attrib_data = buf.data.data() + accessor.byteOffset + buf_view.byteOffset;
            if (attribute.first == "POSITION") {
                //assuming float vec3s bc im lazy FIXME
                glm::vec3* casted_ad = (glm::vec3*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    //transform vertices by tmat
                    vertices.push_back(glm::vec3(tmat * glm::vec4(casted_ad[i], 1.f)));
                }
            }
            else if (attribute.first == "NORMAL") {
                //assuming float vec3
                glm::vec3* casted_ad = (glm::vec3*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    //transform norms by tmat
                    normals.push_back(glm::vec3(tmat * glm::vec4(casted_ad[i], 0.f)));
                }
            }
            //1 set of texture coords per mesh FIXME
            //to fix need to look at refactor structs for more uvs
            else if (attribute.first == "TEXCOORD_0") {
                //assuming vec2 float
                glm::vec2* tex = (glm::vec2*)attrib_data;
                for (int i = 0; i < accessor.count; i++) {
                    uvs.push_back(glm::vec2(glm::mod(tex[i].x, 1.f), glm::mod(tex[i].y, 1.f)));
                }
            }
        }
        
        //tri indices parse
        const tinygltf::Accessor& indices_acc = model.accessors[prim.indices];
        const tinygltf::BufferView& buf_view = model.bufferViews[indices_acc.bufferView];
        tinygltf::Buffer& buf = model.buffers[buf_view.buffer];
        unsigned char* indices_data = buf.data.data() + indices_acc.byteOffset + buf_view.byteOffset;
        //assuming unsigned short or unsigned int indices bc lazy(basically up to 4294967296 vertices) FIXME
        if (indices_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            unsigned short* casted_indices = (unsigned short*)indices_data;
            for (int i = 0; i < indices_acc.count; i++) {
                curr_indices.push_back((unsigned int)(casted_indices[i]));
            }
        }
        else if (indices_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            unsigned int* casted_indices = (unsigned int*)indices_data;
            for (int i = 0; i < indices_acc.count; i++) {
                curr_indices.push_back(casted_indices[i]);
            }
        }

        for (int i = 0; i < vertices.size(); i++) {
            //defaults if no norm/uv for mesh
            glm::vec3 norm = normals.size() > i ? normals[i] : glm::vec3(0);
            glm::vec2 uv = uvs.size() > i ? uvs[i] : glm::vec2(-1);
            MeshPoint mp = { vertices[i], norm, uv };
            curr_mesh_pts.push_back(mp);
        }

        for (int i = 0; i < curr_indices.size() / 3; i++) {
            //construct tri from vertices + mesh index in final arr
            Triangle tri;
            tri.points[0] = curr_mesh_pts[curr_indices[(3 * i)]];
            tri.points[1] = curr_mesh_pts[curr_indices[(3 * i) + 1]];
            tri.points[2] = curr_mesh_pts[curr_indices[(3 * i) + 2]];
            tri.mesh_index = geoms.size() + newGeoms.size();
            curr_tris.push_back(tri);
        }

        mesh_geom.tri_start_index = mesh_triangles.size();
        mesh_triangles.insert(mesh_triangles.end(), curr_tris.begin(), curr_tris.end());
        mesh_geom.tri_end_index = mesh_triangles.size();

        // load mesh texture data + texture if not already loaded
        tinygltf::Material &prim_mat = model.materials[prim.material];
        const auto c = prim_mat.pbrMetallicRoughness.baseColorFactor;
        if (c.size() > 0) {
            mesh_geom.base_color = glm::vec3(c[0], c[1], c[2]);
        }
        tinygltf::TextureInfo& color_tex = prim_mat.pbrMetallicRoughness.baseColorTexture;
        tinygltf::NormalTextureInfo& normal_tex = prim_mat.normalTexture;

        //base color tex
        if (color_tex.index >= 0) {
            tinygltf::Image img = model.images[model.textures[color_tex.index].source];
            //if tex already loaded link it
            if (imguri_to_index.find(img.uri) != imguri_to_index.end()) {
                mesh_geom.texture_index = imguri_to_index[img.uri];
            }
            // if not load it
            else {
                int bytes_p_channel = img.bits / 8;
                std::vector<glm::vec3> rgb_data;
                //img data is 4 channel rgba?
                if (bytes_p_channel == 1) {
                    for (int i = 0; i < img.image.size(); i += 4) {
                        //each byte is 1 channel
                        rgb_data.push_back(glm::vec3((short)(img.image[i]) / 255.f, (short)(img.image[i + 1]) / 255.f, short(img.image[i + 2]) / 255.f));
                    }
                } else if (bytes_p_channel == 2) {
                    unsigned short* casted_2b = (unsigned short*)img.image.data();
                    for (int i = 0; i < img.image.size() / 2; i += 4) {
                        //2 byte is 1 channel
                        rgb_data.push_back(glm::vec3(casted_2b[i] / 65535.f, casted_2b[i + 1] / 65535.f, casted_2b[i + 2] / 65535.f));
                    }
                } else if (bytes_p_channel == 4) {
                    unsigned int* casted_4b = (unsigned int*)img.image.data();
                    for (int i = 0; i < img.image.size() / 4; i += 4) {
                        //4 byte is 1 channel
                        rgb_data.push_back(glm::vec3(casted_4b[i] / 4294967295.f, casted_4b[i + 1] / 4294967295.f, casted_4b[i + 2] / 4294967295.f));
                    }
                }
                // add image info and 
                mesh_geom.texture_index = image_infos.size();
                imguri_to_index.insert({ img.uri, image_infos.size() });
                ImageInfo img_info = { image_data.size(), img.width, img.height };
                image_infos.push_back(img_info);

                image_data.insert(image_data.end(), rgb_data.begin(), rgb_data.end());
            }
        }

        // normal map img 
        if (normal_tex.index >= 0) {
            tinygltf::Image img = model.images[model.textures[normal_tex.index].source];
            //if tex already loaded link it
            if (imguri_to_index.find(img.uri) != imguri_to_index.end()) {
                mesh_geom.texture_index = imguri_to_index[img.uri];
            }
            // if not load it
            else {
                int bytes_p_channel = img.bits / 8;
                std::vector<glm::vec3> normal_data;
                //img data is 4 channel rgba?
                if (bytes_p_channel == 1) {
                    for (int i = 0; i < img.image.size(); i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(img.image[i] / 255.f, img.image[i + 1] / 255.f, img.image[i + 2] / 255.f);
                        // to convert to normals (-0.5), (*2)
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                }
                else if (bytes_p_channel == 2) {
                    unsigned short* casted_2b = (unsigned short*)img.image.data();
                    for (int i = 0; i < img.image.size() / 2; i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(casted_2b[i] / 65535.f, casted_2b[i + 1] / 65535.f, casted_2b[i + 2] / 65535.f);
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                } else if (bytes_p_channel == 4) {
                    unsigned int* casted_4b = (unsigned int*)img.image.data();
                    for (int i = 0; i < img.image.size() / 4; i += 4) {
                        //each byte is 1 channel
                        glm::vec3 rgb = glm::vec3(casted_4b[i] / 4294967295.f, casted_4b[i + 1] / 4294967295.f, casted_4b[i + 2] / 4294967295.f);
                        normal_data.push_back(glm::normalize(2.f * (rgb - 0.5f)));
                    }
                }
                // add image info and 
                mesh_geom.normal_map_index = image_infos.size();
                imguri_to_index.insert({ img.name, image_infos.size() });
                ImageInfo img_info = { image_data.size(), img.width, img.height };
                image_infos.push_back(img_info);

                image_data.insert(image_data.end(), normal_data.begin(), normal_data.end());
            }
        }

        cout << "Created new mesh: " << mesh.name << endl;
        newGeoms.push_back(mesh_geom);
    }
}

void Scene::parseModelNodes(tinygltf::Model& model, tinygltf::Node& node, std::vector<Geom>& newGeoms, glm::mat4 tmat) {
    glm::mat4 cur_tmat = node.matrix.size() == 0 ? glm::mat4(1.f) : glm::make_mat4(node.matrix.data());
    if (node.mesh >= 0 && node.mesh < model.meshes.size()) {
        parseMesh(model, model.meshes[node.mesh], newGeoms, tmat * cur_tmat);
    } 

    for (size_t i = 0; i < node.children.size(); i++) {
        parseModelNodes(model, model.nodes[node.children[i]], newGeoms, tmat * cur_tmat);
    }
}

int Scene::loadGltf(string file, std::vector<Geom> &newGeoms) {
    //gltf loader and model handle + fn
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err, warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, file);
    if (!warn.empty() || !err.empty() || !ret) {
        cout << warn << ", " << err << endl;
        cout << "failed to read gltf file" << endl;
        return -1;
    }

    //one gltf file can produce many Geom mesh objs
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    //iter through all scene nodes for meshes
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        parseModelNodes(model, model.nodes[scene.nodes[i]], newGeoms, glm::mat4(1.f));
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    cout << "Loading Geom " << id << "..." << endl;
    std::vector<Geom> newGeoms;
    string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            Geom newSphere;
            newSphere.type = SPHERE;
            newGeoms.push_back(newSphere);
        } else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            Geom newCube;
            newCube.type = CUBE;
            newGeoms.push_back(newCube);
        // if file ext of str is .gltf load model with tinygltf
        } else if (line.find('.') != std::string::npos && line.substr(line.find_last_of('.') + 1) == "gltf") {
            loadGltf(line, newGeoms);
        } else {
            // bad input
            cerr << "Bad input object :(" << endl;
            exit(-1);
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        for (auto& geom : newGeoms) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            geom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << geom.materialid << "..." << endl;
        }
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            for (auto& geom : newGeoms) {
                geom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            for (auto& geom : newGeoms) {
                geom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            for (auto& geom : newGeoms) {
                geom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        }

        utilityCore::safeGetline(fp_in, line);
    }

    for (auto& geom : newGeoms) {
        geom.transform = utilityCore::buildTransformationMatrix(
            geom.translation, geom.rotation, geom.scale);
        geom.inverseTransform = glm::inverse(geom.transform);
        geom.invTranspose = glm::inverseTranspose(geom.transform);
    }

    //transform all tris to global frame
    for (int i = 0; i < newGeoms.size(); i++) {
        const auto& g = newGeoms[i];
        if (g.type == MESH_PRIM) {
            for (int i = g.tri_start_index; i < g.tri_end_index; i++) {
                Triangle& t = mesh_triangles[i];
                t.points[0].pos = glm::vec3(g.transform * glm::vec4(t.points[0].pos, 1.f));
                t.points[1].pos = glm::vec3(g.transform * glm::vec4(t.points[1].pos, 1.f));
                t.points[2].pos = glm::vec3(g.transform * glm::vec4(t.points[2].pos, 1.f));

                t.points[0].normal = glm::normalize(glm::vec3(g.transform * glm::vec4(t.points[0].normal, 0.f)));
                t.points[1].normal = glm::normalize(glm::vec3(g.transform * glm::vec4(t.points[1].normal, 0.f)));
                t.points[2].normal = glm::normalize(glm::vec3(g.transform * glm::vec4(t.points[2].normal, 0.f)));
            }
        }
    }
    geoms.insert(geoms.end(), newGeoms.begin(), newGeoms.end());
    return 1;
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
