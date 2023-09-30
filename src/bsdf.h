#pragma once

#include "bsdfStruct.h"
#include "texture.h"

__device__ float Lambda(const glm::vec3& w, const float alpha) {
    // Beckmann Spizzichino 
    float theta = acos(w.z);
    float a = 1.0 / (alpha * tan(theta));
    return 0.5f * (erf(a) - 1.0f + exp(-a * a) / (a * sqrt(PI)));
}

__device__ float microfacetBSDF_G(const glm::vec3 & wo, const glm::vec3 & wi, const float alpha) {
    return 1.0 / (1.0 + Lambda(wo, alpha) + Lambda(wi, alpha));
}

__device__ float microfacetBSDF_D(const glm::vec3 & h, const float alpha) {
    float alpha_square = alpha * alpha;
    float cos_theta_h_square = h.z * h.z;
    float tan_theta_h_square = 1.0f / (cos_theta_h_square)-1.0f;
    return exp(-tan_theta_h_square / alpha_square) / (PI * alpha_square * cos_theta_h_square * cos_theta_h_square);
}

template<typename T>
__device__ T Schlick(const T & R0, float cos_theta) {
	return R0 + (1.0f - R0) * pow(1.0f - cos_theta, 5.0f);
}

__device__ glm::vec3 microfacetBSDF_F(const glm::vec3& h, const float ior, const glm::vec3 & baseColor, const float metallic) {
    auto F0 = glm::mix(glm::vec3(0.04f), baseColor, metallic);
    return Schlick(F0, h.z);
    //if (metallic > 0.1f) {
    //    auto F0 = glm::mix(glm::vec3(0.4f), baseColor, metallic);
    //    return Schlick(F0, h.z);
    //}
    //else {
    //    float R0 = powf((1.0f - ior) / (1.0f + ior), 2);
    //    return glm::vec3(Schlick(R0, h.z));

    //}
    //auto eta2_add_k2 = eta * eta + k * k;
    //auto cosine_square = wi.z * wi.z;
    //auto eta_times_cosine = eta * wi.z;
    //auto Rs = (eta2_add_k2 - 2.0f * eta_times_cosine + cosine_square) / ((eta2_add_k2 + 2.0f * eta_times_cosine + cosine_square) + EPSILON);
    //auto Rp = ((eta2_add_k2)*cosine_square - 2.0f * eta_times_cosine + 1.0f) / (((eta2_add_k2)*cosine_square + 2.0f * eta_times_cosine + 1.0f) + EPS_D);
    //return (Rs + Rp) / 2.;
}

__device__ glm::vec3 f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, const glm::vec2 & uv) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
        if (bsdfStruct.baseColorTextureID != -1) {
            return sampleTextureRGB(*bsdfStruct.baseColorTexture, uv) * INV_PI;
		}
		else
			return bsdfStruct.reflectance * INV_PI;
    case MICROFACET:
        glm::vec3 baseColor;
        glm::vec2 metallicRoughness;
        if (bsdfStruct.baseColorTextureID != -1) {
            baseColor = sampleTextureRGB(*bsdfStruct.baseColorTexture, uv);
		}
		else
			baseColor = bsdfStruct.reflectance;
        //baseColor = glm::vec3(1.0f);
        if (bsdfStruct.metallicRoughnessTextureID != -1) {
            metallicRoughness = sampleTextureRG(*bsdfStruct.metallicRoughnessTexture, uv);
        }
        else {
            metallicRoughness = glm::vec2(bsdfStruct.metallicFactor, bsdfStruct.roughnessFactor);
        }
        auto F = microfacetBSDF_F(normalize(wi + wo), bsdfStruct.ior, baseColor, metallicRoughness.x);
        auto alpha = metallicRoughness.y * metallicRoughness.y;
        auto D = microfacetBSDF_D(normalize(wi + wo), alpha);
        auto G = microfacetBSDF_G(wo, wi, alpha);
        auto result = baseColor * F * D * G / (4.0f * wo.z * wi.z);
        //printf("F: %f, D: %f, G: %f\n", F, D, G);
        //printf("result: %f, %f, %f\n", result.x, result.y, result.z);
        return result;
        return baseColor * INV_PI;
    case EMISSIVE:
        return bsdfStruct.emissiveFactor * bsdfStruct.strength;
    default:
        printf("BSDF not implemented!\n");
        assert(0);
        return glm::vec3();
    }
}

__device__ glm::vec3 sample_f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, float* pdf, thrust::default_random_engine & rng, const glm::vec2 & uv) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
    {
        wi = hemiSphereRandomSample(rng, pdf);
        // Encapsulate a sampler class
        return f(bsdfStruct, wo, wi, uv);
    }
    case MICROFACET:
        // Uniform sampling for now
        // TODO: Add importance sampling
        //wi = hemiSphereRandomSample(rng, pdf);
        
        auto s = hemiSphereRandomSample(rng, pdf);
        float alpha = bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor;
        auto theta_h = atan(sqrt(-alpha * alpha * log(1 - s.x)));
        auto phi_h = 2 * PI * s.y;
        glm::vec3 h(sin(theta_h) * cos(phi_h), sin(theta_h) * sin(phi_h), cos(theta_h));
        auto proj = dot(wo, h) * h;
        wi = 2.0f * proj - wo;
        if (wi.z < 0) return glm::vec3();

        auto p_theta = 2 * sin(theta_h) * exp(-pow(tan(theta_h) / alpha, 2)) / (alpha * alpha * pow(cos(theta_h), 3));
        auto p_phi = 1. / (2 * PI);
        auto p_h = p_theta * p_phi / sin(theta_h);
        *pdf = p_h / (4 * dot(wi, h));
        
        return f(bsdfStruct, wo, wi, uv);
        
    case EMISSIVE:
        return f(bsdfStruct, wo, wi, uv);
    default:
        return glm::vec3();
        break;
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