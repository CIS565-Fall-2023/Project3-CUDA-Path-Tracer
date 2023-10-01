#pragma once

#include "sceneStructs.h"

// mat functions
__host__ __device__ void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

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

// microfacet distribution
__host__ __device__ glm::vec3 Sample_wh(glm::vec3 wo, float roughness, thrust::default_random_engine& rng) {
    glm::vec3 wh(0.0f);
    thrust::uniform_real_distribution<float> u01(0, 1);

    float cosTheta = 0;
    float phi = TWO_PI * u01(rng);
    // We'll only handle isotropic microfacet materials

    float xi = u01(rng);
    float tanTheta2 = roughness * roughness * xi / (1.0f - xi);
    cosTheta = 1 / sqrt(1 + tanTheta2);

    float sinTheta =
        sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (wo.z * wh.z <= 0) wh = -wh;

    return wh;
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

// ref: CIS5610 Path Tracer
// TrowbridgeReitz
__host__ __device__ float TrowbridgeReitzD(glm::vec3 wh, float roughness) {
    float cos2Theta = wh.z * wh.z;
    float sinTheta = sqrt(max(0.f, 1.f - cos2Theta));

    float tan2Theta = sinTheta * sinTheta / cos2Theta;
    if (isinf(tan2Theta)) return 0.f;

    float cos4Theta = cos2Theta * cos2Theta;
    
    float cosPhi = sinTheta == 0.0 ? 1.0 : glm::clamp(wh.x / sinTheta, -1.0f, 1.0f);

    float e =
        (cosPhi * cosPhi / (roughness * roughness) 
      + (1.0f - cosPhi * cosPhi) * tan2Theta / (roughness * roughness));
    return 1.0f / (PI * roughness * roughness * cos4Theta * (1.0f + e) * (1.0f + e));
}

__host__ __device__ float Lambda(glm::vec3 w, float roughness) {
    float cosTheta = w.z;
    float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
    float cosPhi = sinTheta == 0.0 ? 1.0 : glm::clamp(w.x / sinTheta, -1.0f, 1.0f);

    float absTanTheta = abs(sinTheta / cosTheta);
    if (isinf(absTanTheta)) return 0.;

    // Compute alpha for direction w
    float alpha =
        sqrt(cosPhi * cosPhi * roughness * roughness + 
        (1.0f - cosPhi * cosPhi) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__host__ __device__ float TrowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness) {
    return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
}

__host__ __device__ glm::vec3 sampleMicrofacetMaterial(
    const Material& m, const glm::vec3& normal,
    const glm::vec3& woW, glm::vec3& wiW, float& pdf, thrust::default_random_engine& rng) {

    glm::vec3 wo = -WorldToLocal(normal) * woW;
    glm::vec3 wh = Sample_wh(wo, m.roughness, rng);
    glm::vec3 wi = glm::reflect(-wo, wh);

    wiW = LocalToWorld(normal) * wi;
    
    if (wo.z * wi.z <= 0.0f) return glm::vec3(0.f);

    float cosThetaO = abs(wo.z);
    float cosThetaI = abs(wi.z);
    wh = glm::normalize(wi + wo);

    if (cosThetaO == 0.0f || cosThetaI == 0.0f || glm::length(wh) == 0.0f) {
        return glm::vec3(0.f);
    }

    float F = 1.0f; // fresnelDielectricEval(dot(wi, wh), 1.0f, m.indexOfRefraction);
    float D = TrowbridgeReitzD(wh, m.roughness);
    float G = TrowbridgeReitzG(wo, wi, m.roughness);

    pdf = D * abs(wh.z) / (4 * dot(wo, wh));

    return m.color * D * F * G / (4.0f * cosThetaO * cosThetaI);
}
