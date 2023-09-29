#pragma once

#include "sceneStructs.h"

__host__ __device__ float fresnelDielectricEval(float cosi, float etai, float etat) {
    if (cosi > 0.0f) {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    cosi = abs(cosi);

    float sint = (etai / etat) * sqrtf(fmaxf(0.0f, 1.0f - cosi * cosi));
    if (sint >= 1.0f) {
        // total internal reflection
        return 1.0;
    }

    float cost = sqrtf(fmaxf(0.0f, 1.0f - sint * sint));

    float Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    float Re = (Rparl * Rparl + Rperp * Rperp) / 2.0f;

    return Re;
}

/*Sample material color*/
__host__ __device__ glm::vec3 sampleSpecularReflectMaterial(
    const Material& m, const glm::vec3& normal, const glm::vec3& wo, glm::vec3& wi) {
    wi = glm::normalize(glm::reflect(wo, normal));
    return m.specular.color / abs(glm::dot(glm::normalize(wi), normal));
}

__host__ __device__ glm::vec3 sampleSpecularTransmissionMaterial(
    const Material& m, const glm::vec3& normal, const glm::vec3& wo, glm::vec3& wi) {
    float etaA = 1.0f, etaB = m.indexOfRefraction;
    float ni = dot(wo, normal);
    glm::vec3 nor = normal;

    bool entering = ni < 0;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    if (!entering) nor = -nor;

    // total internal reflection
    wi = etaI / etaT * sqrtf(fmaxf(0.0f, 1.0f - ni * ni)) > 1.0f ?
        glm::normalize(glm::reflect(wo, nor)) :
        glm::normalize(glm::refract(wo, nor, etaI / etaT));
    return m.specular.color / abs(glm::dot(wi, nor));
}

__host__ __device__ glm::vec3 sampleFresnelSpecularMaterial(
    const Material& m, const glm::vec3& normal,
    const glm::vec3& wo, glm::vec3& wi, thrust::default_random_engine& rng) {

    glm::vec3 nor = normal;
    float cosThetaI = abs(glm::dot(wo, normal));
    float F = fresnelDielectricEval(-cosThetaI, 1.0f, m.indexOfRefraction);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float u = u01(rng);

    if (u < F) {
        // reflect
        return sampleSpecularReflectMaterial(m, normal, wo, wi);
    }
    else {
        // transmission
        return sampleSpecularTransmissionMaterial(m, normal, wo, wi);
    }
}
