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
    
    __device__ virtual Color rho(const glm::vec3& wo, int nSamples, const glm::vec2* samples) const;
    __device__ virtual Color rho(int nSamples, const glm::vec2* samples1, const glm::vec2* samples2) const;

    /* TODO: rho not implemented */
    
    __device__ virtual float pdf(const glm::vec3 & wi, const glm::vec3 & wo) const;

public:
    BxDFType type;
};

class ScaledBxDF : public BxDF {
public:
    ScaledBxDF(BxDF * bxdf, const Color & scale) : BxDF(BxDFType(bxdf->type)), bxdf(bxdf), scale(scale){}
    __device__ Color rho(const glm::vec3 & w, int nSamples, const glm::vec2* samples) const override {
        return scale * bxdf->rho(w, nSamples, samples);
    }

    __device__ Color rho(int nSamples, const glm::vec2* samples1, const glm::vec2* samples2) const override {
        return scale * bxdf->rho(nSamples, samples1, samples2);
    }

    __device__ Color f(const glm::vec3& wo, const glm::vec3& wi) const override {
        return scale * bxdf->f(wo, wi);
    }
    __device__ Color sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& sample, float* pdf, BxDFType* sampledType = nullptr) const override {
        return scale * bxdf->sample_f(wo, wi, sample, pdf, sampledType);
    }

private:
    BxDF* bxdf;
    Color scale;
};

class LambertianReflection: public BxDF{
public:
    LambertianReflection(const Color & R): BxDF(BxDFType(BSDF_REFLECTION|BSDF_DIFFUSE)), R(R){}
    __device__ Color f(const glm::vec3 & wo, const glm::vec3 & wi) const override{
        return R * INV_PI;
    }

    __device__ Color rho(const glm::vec3& w, int nSamples, const glm::vec2* samples) const override {
        return R;
    }

    __device__ Color rho(int nSamples, const glm::vec2* samples1, const glm::vec2* samples2) const override {
        return R;
    }

private:
    const Color R;
};

