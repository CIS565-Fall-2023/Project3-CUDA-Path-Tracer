#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

enum BxDFType{
    BSDF_REFLECTION     = 1<<0,
    BSDF_TRANSMISSION   = 1<<1,
    BSDF_DIFFUSE        = 1<<2,
    BSDF_GLOSSY         = 1<<3,
    BSDF_SPECULAR       = 1<<4,
    BSDF_ALL            = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};

/* May require a better Color class here */
using Color = glm::vec3;

class BxDF{
public:
    virtual ~BxDF(){}
    BxDF(BxDFType type): type(type){}
    __device__ virtual Color f(const glm::vec3 & wo, const glm::vec3 & wi) const = 0;
    __device__ virtual Color sample_f(const glm::vec3 & wo, glm::vec3 * wi, const glm::vec2 & sample, float * pdf, BxDFType * sampledType = nullptr) const;
    __device__ bool matchesFlags(BxDFType t) const{ return (type & t) == type;} 
    
    /* TODO: rho not implemented */
    
    __device__ virtual float pdf(const glm::vec3 & wi, const glm::vec3 & wo) const;

public:
    BxDFType type;
};

class LambertianReflection: public BxDF{
public:
    LambertianReflection(const Color & R): BxDF(BxDFType(BSDF_REFLECTION|BSDF_DIFFUSE)), R(R){}
    __device__ Color f(const glm::vec3 & wo, const glm::vec3 & wi) const override{
        return R * INV_PI;
    }

    __device__ Color sample_f(const glm::vec3 & wo, glm::vec3 * wi, const glm::vec2 & sample, float * pdf, BxDFType * sampledType = nullptr) const override{

    }

    __device__ virtual float pdf(const glm::vec3 & wi, const glm::vec3 & wo) const override;


private:
    const Color R;
};

