#pragma once

#include "bsdfStruct.h"
#include "texture.h"

__device__ float microfacetBSDF_G(const glm::vec3 & wo, const glm::vec3 & wi, const float alpha) {
    return 1.0 / (1.0 + Math::Lambda(wo, alpha) + Math::Lambda(wi, alpha));
}

__device__ float microfacetBSDF_D(const glm::vec3 & h, const float alpha) {
    float alpha_square = glm::max(alpha * alpha, EPSILON);
    auto tan2Theta = Math::Tan2Theta(h);
    if (glm::isinf(tan2Theta)) return 0.;
    auto cos4Theta = glm::max(Math::Cos2Theta(h) * Math::Cos2Theta(h), EPSILON);
	return exp(-tan2Theta / alpha_square) 
        / (PI * alpha_square * cos4Theta);
}

template<typename T>
__device__ T Schlick(const T & R0, float cos_theta) {
    auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
    return R0 + (1.0f - R0) * pow5(1.0f - cos_theta);
}

__device__ float roughnessToAlpha(float roughness) {
    roughness = __max(roughness, 1e-3);
    float x = glm::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
        0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

__device__ glm::vec3 microfacetBSDF_F(const glm::vec3& h, const glm::vec3 & wo, const glm::vec3 & baseColor, const float metallic) {
    auto F0 = glm::mix(glm::vec3(0.08f), baseColor, metallic);
    return Schlick(F0, glm::dot(wo, h));
}

__device__ glm::vec3 f(BSDFStruct& bsdfStruct, const glm::vec3& wo, const glm::vec3& wi, const glm::vec2 & uv) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
		return bsdfStruct.reflectance * INV_PI;
    case MICROFACET:
        glm::vec3 baseColor;
        glm::vec2 metallicRoughness;
		baseColor = bsdfStruct.reflectance;
        metallicRoughness = glm::vec2(bsdfStruct.metallicFactor, bsdfStruct.roughnessFactor);
        //auto alpha = roughnessToAlpha(metallicRoughness.y);
        auto alpha = metallicRoughness.y * metallicRoughness.y;
        auto h = glm::normalize(wi + wo);
        auto D = microfacetBSDF_D(h, alpha);
        auto G = microfacetBSDF_G(wo, wi, alpha);
        auto F = microfacetBSDF_F(h, wo, baseColor, metallicRoughness.x);
        auto denominator = 4.0f * wo.z * wi.z + EPSILON;
        auto diffuse = baseColor * INV_PI * (1.f - metallicRoughness.x);
        auto specular = glm::vec3(G * D / denominator);
        return glm::mix(diffuse, specular, F);
    case EMISSIVE:
        return bsdfStruct.emissiveFactor * bsdfStruct.strength;
    default:
        printf("BSDF not implemented!\n");
        assert(0);
        return glm::vec3();
    }
}

__device__ float PDF(BSDFStruct& bsdfStruct, const glm::vec3& wo, const glm::vec3& wi) {
    switch (bsdfStruct.bsdfType) {
        case DIFFUSE:
		    return abs(wi.z) * INV_PI;
        case MICROFACET:
            auto wh = glm::normalize(wi + wo);
            float alpha = bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor;
            float denominator = glm::max(4 * glm::dot(wo, wh), EPSILON);
            return glm::mix(abs(wi.z) * INV_PI, microfacetBSDF_D(wh, alpha) * wh.z / denominator, 1.f / (2.f - bsdfStruct.metallicFactor));
        default:
            break;
    }
}

__device__ glm::vec3 sample_f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, float* pdf, thrust::default_random_engine & rng, const glm::vec2 & uv, const glm::vec3 & _rands) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
    {
        thrust::random::uniform_real_distribution<float> u01(0, 1);
        wi = hemiSphereRandomSample(glm::vec2(u01(rng), u01(rng)), pdf);
        //wi = hemiSphereRandomSample(glm::vec2(rands), pdf);
        // Encapsulate a sampler class
        return f(bsdfStruct, wo, wi, uv);
    }
    case MICROFACET:
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 rands(u01(rng), u01(rng), u01(rng));
        float alpha = glm::max(bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor, EPSILON);
        glm::vec3 wi_diffuse;
        float pdf_diffuse;
        //wi_diffuse = hemiSphereRandomSample(glm::vec2(rands), &pdf_diffuse);
        glm::vec3 wh;
        if (rands.z > (1.f / (2.f - bsdfStruct.metallicFactor))) {
            wi = hemiSphereRandomSample(glm::vec2(rands), &pdf_diffuse);
            wh = glm::normalize(wi + wo);
            //printf("Diffuse wi.z: %f\n", wi.z);
        }
        else {
            glm::vec2 u(rands);
            //float alpha = roughnessToAlpha(bsdfStruct.roughnessFactor);
            float logSample = glm::log(1 - u[0]);
            if (glm::isinf(logSample)) logSample = 0;
            float tanTheta2 = -alpha * alpha * logSample;
            float phi = u[1] * 2 * PI;
            float cosTheta = 1 / sqrt(1 + tanTheta2);
            float sinTheta = sqrt(glm::max(0.0f, 1 - cosTheta * cosTheta));
            wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
            auto d = glm::dot(wo, wh);
            *pdf = 1.0f;
            if (d <= EPSILON) return glm::vec3();
            wi = -wo + 2.0f * d * wh;
            //wi = glm::reflect(-wo, wh);
            //printf("Specular wi.z: %f wo: %f %f %f wh : %f %f %f\n", wi.z, wo.x, wo.y, wo.z, wh.x, wh.y, wh.z);
        }
        *pdf = PDF(bsdfStruct, wo, wi);
        return f(bsdfStruct, wo, wi, uv);
        
    case EMISSIVE:
        return f(bsdfStruct, wo, wi, uv);
    default:
        return glm::vec3();
        break;
    }
}

__device__ void initBSDF(BSDFStruct& bsdfStruct, const glm::vec2 & uv) {
    if (bsdfStruct.baseColorTextureID != -1) {
        bsdfStruct.reflectance = sampleTextureRGB(*bsdfStruct.baseColorTexture, uv);
    }

    glm::vec2 metallicRoughness;
    if (bsdfStruct.metallicRoughnessTextureID != -1) {
        metallicRoughness = sampleTextureRG(*bsdfStruct.metallicRoughnessTexture, uv);
        bsdfStruct.metallicFactor = glm::max(metallicRoughness.x, EPSILON);
        bsdfStruct.roughnessFactor = glm::max(metallicRoughness.y, EPSILON);
    }
}

__device__ glm::vec3 get_debug_color(BSDFStruct& bsdfStruct) {
	switch (bsdfStruct.bsdfType)
	{
	case DIFFUSE:
		return glm::vec3(1.0f, 0.0f, 0.0f);
	case EMISSIVE:
		return glm::vec3(0.0f, 1.0f, 0.0f);
	default:
		return glm::vec3(0.0f, 0.0f, 1.0f);
	}
}