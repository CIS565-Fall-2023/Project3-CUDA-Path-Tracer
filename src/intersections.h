#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define AABB_MESH_CULLING 1

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// tests if a ray intersects with a bounding box
// adapted from https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/
__host__ __device__ float aabbRayIntTest(glm::vec3 bmin, glm::vec3 bmax, Ray r, float max_dist) {
    float tx1 = (bmin.x - r.origin.x) / r.direction.x, tx2 = (bmax.x - r.origin.x) / r.direction.x;
    float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
    float ty1 = (bmin.y - r.origin.y) / r.direction.y, ty2 = (bmax.y - r.origin.y) / r.direction.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - r.origin.z) / r.direction.z, tz2 = (bmax.z - r.origin.z) / r.direction.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    if (tmax >= tmin && tmin < max_dist && tmax > 0) return tmin; else return 1e30;
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r, glm::vec3& intPt, glm::vec3& normal, bool& outside, 
    Triangle* triangles, ImageInfo* img_info, glm::vec3* img_data, glm::vec3& bary, int &tri_index) {
    // bring ray into frame of mesh
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    //bb int culling
    // check against mesh bb, if no int return
#if AABB_MESH_CULLING
    // as mesh bb in mesh frame use ray in mesh frame
    if (aabbRayIntTest(mesh.bb_min, mesh.bb_max, q, 1e30) == 1e30) return -1;
#endif

    //check against each tri of mesh with glm::intersectRayTriangle
    //find smallest t to return(int pt)
    //list tri normal by bary interp verts
    float t_min = FLT_MAX;
    Triangle* min_tri_ptr;
    int min_tri_index = -1;
    glm::vec3 int_bary;
    for (int i = mesh.tri_start_index; i < mesh.tri_end_index; i++) {
        Triangle& curr_tri = triangles[i];
        //res is (v2r, v3r, dist to int)?
        glm::vec3 curr_bary;
        if (glm::intersectRayTriangle(q.origin, q.direction, curr_tri.points[0].pos, curr_tri.points[1].pos, curr_tri.points[2].pos, curr_bary)) {
            float cur_t = curr_bary.z;
            if (cur_t < t_min) {
                min_tri_ptr = &triangles[i];
                t_min = cur_t;
                int_bary = curr_bary;
                min_tri_index = i;
            }
        }
    }

    // if no intersections return
    if (t_min == FLT_MAX) return -1;

    //normal based of following priorities
    // 1. normal map
    // 2. bary interp normal gltf info
    // 3. triangle flat normal
    glm::vec3 tri_normal;
    int_bary.z = (1.f - int_bary.x - int_bary.y);
    Triangle min_tri = *min_tri_ptr;
    if (mesh.normal_map_index >= 0) {
        //bary interp uv
        glm::vec2 mesh_uv = int_bary.z * min_tri.points[0].tex_uv + int_bary.x * min_tri.points[1].tex_uv + int_bary.y * min_tri.points[2].tex_uv;
        //sample data for correct normal
        ImageInfo normal_map = img_info[mesh.normal_map_index];
        glm::ivec2 sample_uv = mesh_uv * glm::vec2(normal_map.img_w, normal_map.img_h);
        tri_normal = img_data[normal_map.data_start_index + (sample_uv[1] * normal_map.img_w + sample_uv[0])];
    } 
    else if (min_tri.points[0].normal != glm::vec3(0)) {
        tri_normal = (1.f - int_bary.x - int_bary.y) * min_tri.points[0].normal + int_bary.x * min_tri.points[1].normal + int_bary.y * min_tri.points[2].normal;
    }
    else {
        tri_normal = outside ? glm::cross(min_tri.points[2].pos - min_tri.points[0].pos, min_tri.points[1].pos - min_tri.points[0].pos) : glm::cross(min_tri.points[1].pos - min_tri.points[0].pos, min_tri.points[2].pos - min_tri.points[0].pos);
    }
    //convert back to global frame
    intPt = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, t_min), 1.f));
    normal = glm::normalize(multiplyMV(mesh.transform, glm::vec4(tri_normal, 0.f)));
    bary = int_bary;
    tri_index = min_tri_index;
    return glm::length(r.origin - intPt);
}

#if BVH
//bvh intersection test
//for a given ray traverses the scene bvh tree to find closest int pt if any
// this could be both on a sphere/cube or a mesh tri
// FIXME there are an unholy number of parameters to this function
__host__ __device__ float bvh_intersection_test(Ray r, glm::vec3& int_pt, glm::vec3& normal, bool& outside,
    Triangle* triangles, Geom* geoms, ImageInfo* img_info, glm::vec3* img_data, glm::vec3& bary, int& tri_index, int& geom_index, //-1 for one
    BVHNode* bvh_tree, BVHTriIndex* bvh_tri_indices, BVHGeomIndex* bvh_geom_indices) {
    //stack based traversal from same bvh blog post
    //root at index 0
    BVHNode* stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = &bvh_tree[0]; //start at root
    float min_t = 1e30;
    while (stack_ptr > 0) {
        BVHNode* node = stack[--stack_ptr];
        // if this bb does not intersect or intersects farther
        // than found already skip
        if (aabbRayIntTest(node->min, node->max, r, min_t) >= min) continue;
        // no left child = a leaf
        if (node->leftNode == -1) {
            //test ray intersection on each prim at this leaf
            for (int i = 0; i < node->triCount; i++) {
                int triangle_index = bvh_tri_indices[node->triIndexStart + i].triIndex;
                Triangle& tri = triangles[triangle_index];
                Geom& mesh = geoms[tri.mesh_index];
                glm::vec3 cur_bary;
                // move tri points to global frame
                Ray q;
                q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
                q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

                if (glm::intersectRayTriangle(q.origin, q.direction, tri.points[0].pos, tri.points[1].pos, tri.points[2].pos, cur_bary)) {
                    int_pt = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, cur_bary[2]), 1.f));
                    float cur_t = glm::length(r.origin - int_pt);
                    if (cur_t < min_t) {
                        min_t = cur_t;
                        //update int_pt, normal, outside, bary, geom_index, tri_index
                        cur_bary.z = 1 - cur_bary.x - cur_bary.y;
                        bary = cur_bary;
                        outside = false;
                        tri_index = triangle_index;
                        geom_index = tri.mesh_index;
                    }
                }
            }

            for (int i = 0; i < node->geomCount; i++) {
                Geom& g = geoms[bvh_geom_indices[node->geomIndexStart + i].geomIndex];
                //call the sphere/cube int methods
                //these auto move ray to geom frame so ok
                if (g.type == CUBE) {
                    float cur_t = boxIntersectionTest(g, r, int_pt, normal, outside);
                    if (cur_t == -1) cur_t = 1e30;
                    if (cur_t < min_t) {
                        min_t = cur_t;
                        geom_index = bvh_geom_indices[node->geomIndexStart + i].geomIndex;
                        tri_index = -1;
                    }
                }
                else if (g.type == SPHERE) {
                    float cur_t = sphereIntersectionTest(g, r, int_pt, normal, outside);
                    if (cur_t == -1) cur_t = 1e30;
                    if (cur_t < min_t) {
                        min_t = cur_t;
                        geom_index = bvh_geom_indices[node->geomIndexStart + i].geomIndex;
                        tri_index = -1;
                    }
                }
            }
            continue;
        }
        stack[stack_ptr++] = &bvh_tree[node->leftNode];
        stack[stack_ptr++] = &bvh_tree[node->leftNode + 1];
    }
    //return if any intersection(this should be closest int)
    if (geom_index >= 0) {
        //if tri intersection do normal interp
        //FIXME bring to global frame
        if (tri_index >= 0) {
            Triangle& tri = triangles[tri_index];
            Geom& mesh = geoms[tri.mesh_index];
            if (mesh.normal_map_index >= 0) {
                //bary interp uv
                glm::vec2 mesh_uv = bary.z * tri.points[0].tex_uv + bary.x * tri.points[1].tex_uv + bary.y * tri.points[2].tex_uv;
                //sample data for correct normal
                ImageInfo normal_map = img_info[mesh.normal_map_index];
                glm::ivec2 sample_uv = mesh_uv * glm::vec2(normal_map.img_w, normal_map.img_h);
                normal = img_data[normal_map.data_start_index + (sample_uv[1] * normal_map.img_w + sample_uv[0])];
            }
            else if (tri.points[0].normal != glm::vec3(0)) {
                normal = (1.f - bary.x - bary.y) * tri.points[0].normal + bary.x * tri.points[1].normal + bary.y * tri.points[2].normal;
            }
            else {
                normal = outside ? glm::cross(tri.points[2].pos - tri.points[0].pos, tri.points[1].pos - tri.points[0].pos) : glm::cross(tri.points[1].pos - tri.points[0].pos, tri.points[2].pos - tri.points[0].pos);
            }
            normal = glm::normalize(multiplyMV(mesh.transform, glm::vec4(normal, 0.f)));
        }
        return min_t;
    }
}

#endif