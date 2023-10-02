#pragma once

#include "sceneStructs.h"

// sample lights
__host__ __device__ glm::vec3 sampleAreaLight(const Light& light, const glm::vec3& view_point, const glm::vec3& view_nor,
    int num_lights, glm::vec3& wiW, float& pdf, 
    Material& mat, thrust::default_random_engine& rng) {

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
        pdf = r * r / (costheta * light.geom.scale.x * light.geom.scale.z);

        // Set ¦Øi to the normalized vector from the reference point
        // to the generated light source point
        wiW = normalize(dis);

        // test visibility outside
        return mat.color * mat.emittance / (float)num_lights;
    }

    return glm::vec3(0.0);
}