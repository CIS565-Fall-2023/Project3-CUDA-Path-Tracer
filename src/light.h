#pragma once

#include "sceneStructs.h"

__host__ __device__ glm::vec3 getEnvLight(const glm::vec3* envMap, int width, int height, glm::vec3 wiW, int num_lights) {
    glm::vec2 uv = glm::vec2(glm::atan(wiW.z, wiW.x) / TWO_PI, asin(wiW.y) / PI);
    uv += glm::vec2(0.5f);
    uv *= glm::vec2(width, height);
    int index = (int)uv.y * width + (int)uv.x;

    return index < width * height ? envMap[index] * (float)num_lights : glm::vec3(0.0f);
}

__host__ __device__ glm::vec3 sampleEnvLight(const glm::vec3* envMap, int width, int height, const glm::vec3& view_nor,
    int num_lights, glm::vec3& wiW, float& pdf, thrust::default_random_engine& rng) {

    glm::vec3 d = squareToDiskConcentric(rng);
    glm::vec3 wi = glm::vec3(d.x, d.y, sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y)));

    wiW = LocalToWorld(view_nor) * wi;
    pdf = wi.z / PI;

    return getEnvLight(envMap, width, height, wiW, num_lights);
}

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
        return mat.color * mat.emittance * (float)num_lights;
    }

    return glm::vec3(0.0);
}


__host__ __device__ glm::vec3 sampleLight(
      glm::vec3 point
    , glm::vec3 normal
    , glm::vec3& wiW
    , Material* materials
    , Light* lights
    , int numLights
    , LightType& chosenLightType
    , float& lightPdf
    , int& chosenLightGeomId
    , glm::vec3* envMap
    , int width
    , int height
    , thrust::default_random_engine& rng) {

    glm::vec3 Li(0.0f);

    thrust::uniform_int_distribution<int> u_light(0, numLights - 1);
    int chosenLightIndex = u_light(rng);

    if (envMap != NULL && chosenLightIndex == numLights - 1) {
        // sample enviorment light
        chosenLightType = LightType::ENVIRONMENT;
        chosenLightGeomId = -1;
        Li = sampleEnvLight(envMap, width, height, normal, numLights, wiW, lightPdf, rng);
    }
    else {
        Light chosenLight = lights[chosenLightIndex];
        chosenLightType = chosenLight.lightType;
        chosenLightGeomId = chosenLight.geom.geomId;

        switch (chosenLight.lightType)
        {
            case LightType::AREA:
            {
                Li = sampleAreaLight(chosenLight, point, normal, numLights, wiW, lightPdf,
                    materials[chosenLight.geom.materialid], rng);
                break;
            }
        }
    }

    return Li;
}
