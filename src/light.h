#pragma once

#include "sceneStructs.h"

// sample lights
__host__ __device__ glm::vec3 sampleAreaLight(const Light& light, glm::vec3& view_point, glm::vec3& view_nor,
    int num_lights, glm::vec3& wiW, float& pdf, Geom* geoms, int geom_size, Material* materials, thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    switch (light.geom.type)
    {
    case CUBE:
        glm::mat4 t = light.geom.transform;
        glm::vec4 sample = glm::vec4(u01(rng) - 0.5f, 0.0f, u01(rng) - 0.5f, 1.0f);
        glm::vec4 light_point_W = t * sample;
        glm::vec3 light_nor_W = glm::normalize(multiplyMV(light.geom.invTranspose, glm::vec4(0., 1., 0., 0.)));

        // Compute and Convert Pdf
        glm::vec3 dis = glm::vec3(light_point_W) - view_point;
        float r = length(dis);
        float costheta = abs(dot(-normalize(dis), light_nor_W));
        pdf = r * r / (costheta * light.geom.scale.x * light.geom.scale.z * (float)num_lights);

        // Set ¦Øi to the normalized vector from the reference point
        // to the generated light source point
        wiW = normalize(dis);

        // Check to see if ¦Øi reaches the light source
        Ray shadowRay;
        shadowRay.origin = view_point + 0.01f * wiW;
        shadowRay.direction = wiW;

        ShadeableIntersection intersection;

        computeRayIntersection(geoms, geom_size, shadowRay, intersection);

        if (intersection.t >= 0.0f && intersection.geomId == light.geom.geomId) {
            return materials[intersection.materialId].color * materials[intersection.materialId].emittance;
        }
    }

    return glm::vec3(0.0);
}