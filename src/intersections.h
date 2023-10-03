#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define DEBUG 1

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


__host__ __device__ inline float intersect_aabb(const glm::vec3& ray_origin, const glm::vec3& ray_rec_direction, const Aabb& aa_bb, int curr_t)
{
    glm::vec3 bmin = aa_bb.bmin;
    glm::vec3 bmax = aa_bb.bmax;
    float tx_1 = (bmin.x - ray_origin.x) * ray_rec_direction.x;
    float tx_2 = (bmax.x - ray_origin.x) * ray_rec_direction.x;
    float t_min = min(tx_1, tx_2), tmax = max(tx_1, tx_2);
    float ty_1 = (bmin.y - ray_origin.y) * ray_rec_direction.y;
    float ty_2 = (bmax.y - ray_origin.y) * ray_rec_direction.y;
    t_min = max(t_min, min(ty_1, ty_2)), tmax = min(tmax, max(ty_1, ty_2));
    float tz_1 = (bmin.z - ray_origin.z) * ray_rec_direction.z;
    float tz_2 = (bmax.z - ray_origin.z) * ray_rec_direction.z;
    t_min = max(t_min, min(tz_1, tz_2)), tmax = min(tmax, max(tz_1, tz_2));

    return (tmax >= t_min && t_min < curr_t && tmax > 0) ? t_min : FLT_MAX;
}


// returns t
__host__ __device__ float intersect_bvh(const Geom &geom, Triangle *tris, const Ray &ray, glm::vec3& intersect, 
                                        glm::vec3& normal, BvhNode *bvh_nodes, int root_node_index)
{
    BvhNode* node = &bvh_nodes[root_node_index];
    int stack[64];
    int stack_ptr = 0;
    Ray r = ray;
    r.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    r.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    glm::vec3 ray_origin = r.origin;
    glm::vec3 ray_rec_direction = 1.0f / r.direction;
    glm::vec3 ray_direction = r.direction;
    float curr_t = FLT_MAX; //Double check what this does
    do
    {
        if (node->tri_count > 0) // if BvhNode is leaf
        {
            for (unsigned int i = 0; i < node->tri_count; i++)
            {
                // get intersection with each triangle
                Triangle* tri = &tris[node->left_first + i];
                glm::vec3 tri_v0 = tri->v0.pos;
                glm::vec3 tri_v1 = tri->v1.pos;
                glm::vec3 tri_v2 = tri->v2.pos;
                glm::vec3 bary;

                
                // bary.z might be t
                if (glm::intersectRayTriangle(ray_origin, ray_direction, tri_v0, tri_v1, tri_v2, bary) 
                    && (bary.z < curr_t))
                {
                    curr_t = bary.z;
                    float bary_w = 1.0f - bary.x - bary.y;
                    intersect = getPointOnRay(r, curr_t);
#if DEBUG
                    // I'm not sure if this is in world coords
                    glm::vec3 intersect_test = tri_v0 * bary_w + tri_v1 * bary.x + tri_v2 * bary.y; 
                    glm::vec3 diff = intersect - intersect_test;
                    printf("diff is %f, %f, %f", diff.x, diff.y, diff.z);
#endif DEBUG
                    // I'm not sure if this is in world coords
                    normal = glm::normalize(tri->v0.nor * bary_w + tri->v1.nor * bary.x + tri->v2.nor * bary.y);

                    //intersect = bary.x * tri_v0 + bary.y * tri_v1 + bary_w * tri_v2; //try w = 1 - (u + v) here
                    //normal = glm::normalize(bary.x * tri->v0.nor + bary.y * tri->v1.nor + bary_w * tri->v2.nor);
                }

            }
            if (stack_ptr == 0)
            {
				break;
			}
            node = &bvh_nodes[stack[--stack_ptr]]; // Double check this logic for first iter
        }
        else
        {
            int child_1_index = node->left_first;
            int child_2_index = node->left_first + 1;
            // Get distances to intersections
            float dist_1 = intersect_aabb(ray_origin, ray_rec_direction, bvh_nodes[child_1_index].aa_bb, curr_t);
            float dist_2 = intersect_aabb(ray_origin, ray_rec_direction, bvh_nodes[child_2_index].aa_bb, curr_t);
            // Check closer node first
            if (dist_1 > dist_2)
            {
                // swap
                float temp = dist_1;
                dist_1 = dist_2;
                dist_2 = temp;
                temp = child_1_index;
                child_1_index = child_2_index;
                child_2_index = temp;
            }
            if (dist_1 == FLT_MAX)
            {
                // If no intersection, continue popping off stack
                if (stack_ptr == 0)
                {
                    break;
                }
                node = &bvh_nodes[stack[--stack_ptr]];
            }
            else
            {
                // If valid intersection, add children to stack
                node = &bvh_nodes[child_1_index];
                if (dist_2 < FLT_MAX)
                {
                    stack[stack_ptr++] = child_2_index;
                }
            }
        }
    } while (stack_ptr != 0); //double check iterations
    intersect = multiplyMV(geom.transform, glm::vec4(intersect, 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.0f)));
#if DEBUG
    float length = glm::length(r.origin - intersect);
    float distance =
        printf("curr_t is %f, length is %f", curr_t, length);
#endif DEBUG
    return glm::length(ray.origin - intersect); // return distance
}
