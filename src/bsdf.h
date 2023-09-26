#include "bsdfStruct.h"

__device__ glm::vec3 f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
        return bsdfStruct.reflectance / PI;
    case EMISSIVE:
        return bsdfStruct.emissiveFactor * bsdfStruct.strength;
    default:
        printf("BSDF not implemented!\n");
        assert(0);
        return glm::vec3();
    }
}

__device__ glm::vec3 sample_f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, float* pdf) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
    {
        // Encapsulate a sampler class
        return bsdfStruct.reflectance / PI;
    }
    case EMISSIVE:
        return f(bsdfStruct, wo, wi);
    default:
        return glm::vec3();
        break;
    }
}