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

// tests if a ray intersects with a bounding box in global frame
// adapted from https://gdbooks.gitbooks.io/3dcollisions/content/Chapter3/raycast_aabb.html
__host__ __device__ float aabbRayIntTest(glm::vec3 bb_min, glm::vec3 bb_max, Ray r) {
    float t1 = (bb_min.x - r.origin.x) / r.direction.x;
    float t2 = (bb_max.x - r.origin.x) / r.direction.x;
    float t3 = (bb_min.y - r.origin.y) / r.direction.y;
    float t4 = (bb_max.y - r.origin.y) / r.direction.y;
    float t5 = (bb_min.z - r.origin.z) / r.direction.z;
    float t6 = (bb_max.z - r.origin.z) / r.direction.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us
    if (tmax < 0) {
        return -1;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax) {
        return -1;
    }

    if (tmin < 0.f) {
        return tmax;
    }
    return tmin;
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
    if (aabbRayIntTest(mesh.bb_min, mesh.bb_max, q) == -1) return -1;
#endif

    //check against each tri of mesh with glm::intersectRayTriangle
    //find smallest t to return(int pt)
    //list tri normal by bary interp verts
    float t_min = FLT_MAX;
    Triangle* min_tri_ptr;
    int min_tri_index;
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