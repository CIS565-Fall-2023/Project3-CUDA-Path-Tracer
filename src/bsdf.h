#pragma once
#include "sceneStructs.h"
#include <glm/glm.hpp>
#include <thrust/random.h>


enum BSDFType {
    UNIMPLEMENTED = -1,
    DIFFUSE = 0,
    SPECULAR = 1,
    REFRACTIVE = 2,
    MICROFACET = 3,
    EMISSIVE = 4
};

class BSDFStruct {
public:
    glm::vec3 reflectance;
    glm::vec3 emissiveFactor;
    float strength;
    BSDFType bsdfType;
};



__device__ glm::vec3 f(BSDFStruct & bsdfStruct, const glm::vec3& wo, glm::vec3& wi) {
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
        break;
    }
}