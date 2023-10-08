#pragma once

#include <glm/glm.hpp>
#include "common.h"
#include "sceneStructs.h"

class Triangle
{
public:
    /**
     * Test intersection between a ray and triangle.
     *
     * @param ray                Input parameter of testing ray
     * @param index              3 unsigned int value indicate position of vertices, normals, and uvs
     * @param vertices           vertices array
     * @param normals            normals array
     * @param uvs                uvs array
     * @param
     * @return                   true or false, indicate whether intersect with triangle
     */
    CPU_GPU static bool Intersection(const Ray& ray,
                                     const glm::vec3* v,
                                     Intersection& intersection)
    {
        // Moller¡§CTrumbore intersection methods

        glm::vec3 e01 = v[1] - v[0];
        glm::vec3 e02 = v[2] - v[0];

        glm::vec3 p_vec = glm::cross(ray.direction, e02);

        float det = glm::dot(p_vec, e01);

        // ray is perpendicular to the plane contains the triangle
        if (glm::abs(det) < Epsilon) return false;

        glm::vec3 t_vec = ray.origin - v[0];

        if (det < 0.f)
        {
            det = -det;
            t_vec = -t_vec;
        }

        glm::vec2 uv; // local uv in triangle

        uv.x = glm::dot(t_vec, p_vec);

        if (uv.x < 0.f || uv.x > det) return false;

        glm::vec3 q_vec = glm::cross(t_vec, e01);

        uv.y = glm::dot(ray.direction, q_vec);

        if (uv.y < 0.f || uv.x + uv.y > det) return false;

        float inv_det = 1.f / det;
        uv *= inv_det;

        float t = glm::dot(e02, q_vec) * inv_det;
        intersection.t = t;
        intersection.uv = uv;

        return t > 0.f;
    }
};