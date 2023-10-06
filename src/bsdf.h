#pragma once

#include "bsdfStruct.h"
#include "texture.h"

#define Clamp(x, min, max) (x < min ? min : (x > max ? max : x))

// BSDF Inline Functions
__device__ inline float CosTheta(const glm::vec3& w) { return w.z; }
__device__ inline float Cos2Theta(const glm::vec3& w) { return w.z * w.z; }
__device__ inline float AbsCosTheta(const glm::vec3& w) { return std::abs(w.z); }
__device__ inline float Sin2Theta(const glm::vec3& w) {
    return __max((float)0, (float)1 - Cos2Theta(w));
}

__device__ inline float SinTheta(const glm::vec3& w) { return std::sqrt(Sin2Theta(w)); }

__device__ inline float TanTheta(const glm::vec3& w) { return SinTheta(w) / CosTheta(w); }

__device__ inline float Tan2Theta(const glm::vec3& w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

__device__ inline float CosPhi(const glm::vec3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}

__device__ inline float SinPhi(const glm::vec3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

__device__ inline float Cos2Phi(const glm::vec3& w) { return CosPhi(w) * CosPhi(w); }

__device__ inline float Sin2Phi(const glm::vec3& w) { return SinPhi(w) * SinPhi(w); }

__device__ inline float CosDPhi(const glm::vec3& wa, const glm::vec3& wb) {
    float waxy = wa.x * wa.x + wa.y * wa.y;
    float wbxy = wb.x * wb.x + wb.y * wb.y;
    if (waxy == 0 || wbxy == 0)
        return 1;
    return Clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1, 1);
}


__device__ float Lambda(const glm::vec3& w, const float alpha) {
    float absTanTheta = std::abs(TanTheta(w));
    if (glm::isinf(absTanTheta)) return 0.;
        float alpha_lambda = glm::sqrt(Cos2Phi(w) +
            Sin2Phi(w)) * alpha;

    float a = 1 / (alpha_lambda * absTanTheta);
    if (a >= 1.6f)
        return 0;
    return (1 - 1.259f * a + 0.396f * a * a) /
        (3.535f * a + 2.181f * a * a);
    
    // Beckmann Spizzichino 
    //float theta = acos(w.z);
    //float a = 1.0 / (alpha * tan(theta));
    //return 0.5f * (erf(a) - 1.0f + exp(-a * a) / (a * sqrt(PI)));
}

__device__ float microfacetBSDF_G(const glm::vec3 & wo, const glm::vec3 & wi, const float alpha) {
    return 1.0 / (1.0 + Lambda(wo, alpha) + Lambda(wi, alpha));
}

__device__ float microfacetBSDF_D(const glm::vec3 & h, const float alpha) {
    float alpha_square = alpha * alpha;
    auto tan2Theta = Tan2Theta(h);
    if (glm::isinf(tan2Theta)) return 0.;
    auto cos4Theta = Cos2Theta(h) * Cos2Theta(h);
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

        if (bsdfStruct.metallicRoughnessTextureID != -1) {
            metallicRoughness = sampleTextureRG(*bsdfStruct.metallicRoughnessTexture, uv);
        }
        else {
            metallicRoughness = glm::vec2(bsdfStruct.metallicFactor, bsdfStruct.roughnessFactor);
        }
        //auto alpha = roughnessToAlpha(metallicRoughness.y);
        auto alpha = metallicRoughness.y * metallicRoughness.y;
        auto h = glm::normalize(wi + wo);
        auto D = microfacetBSDF_D(h, alpha);
        auto G = microfacetBSDF_G(wo, wi, alpha);
        auto F = microfacetBSDF_F(h, wo, baseColor, metallicRoughness.x);
        //if (metallicRoughness.x < 0.99f) printf("h: %f %f %f kS: %f %f %f kD:%f %f %f metallic:%f\n", normalize(wi + wo).x, normalize(wi + wo).y, normalize(wi + wo).z, kS.x, kS.y, kS.z, kD.x, kD.y, kD.z, metallicRoughness.x);
        auto denominator = 4.0f * wo.z * wi.z + EPSILON;
        //auto specular = F * D * G / denominator;
        //auto kS = metallicRoughness.x;
        //auto kD = 1.0f - kS;
        //auto specular = kS * F * D * G / denominator;
        //auto diffuse  = kD * INV_PI;
        //auto result = baseColor * (diffuse + specular);
        //auto result = baseColor * specular;
        //auto result = baseColor * diffuse;
        //return (baseColor * F* D* G/denominator);
        //return baseColor * INV_PI;
        auto diffuse = baseColor * INV_PI * (1.f - metallicRoughness.x);
        auto specular = glm::vec3(G * D / denominator);
        return glm::mix(diffuse, specular, F);
        //printf("baseColor: %f %f %f\n", baseColor.x, baseColor.y, baseColor.z);

        //return baseColor;
        //return result;
        //return baseColor * INV_PI;
    case EMISSIVE:
        return bsdfStruct.emissiveFactor * bsdfStruct.strength;
    default:
        printf("BSDF not implemented!\n");
        assert(0);
        return glm::vec3();
    }
}

__device__ float PDF(BSDFStruct& bsdfStruct, const glm::vec3& wo, const glm::vec3& wi, const glm::vec3 & rands) {
    switch (bsdfStruct.bsdfType) {
        case DIFFUSE:
		    return abs(wi.z) * INV_PI;
        case MICROFACET:
            auto wh = glm::normalize(wi + wo);
            float alpha = bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor;
            return glm::mix(abs(wi.z) * INV_PI, microfacetBSDF_D(wh, alpha) * wh.z / (4 * glm::dot(wo, wh)), 1.f / (2.f - bsdfStruct.metallicFactor));
        default:
            break;
    }
}

__device__ glm::vec3 sample_f(BSDFStruct& bsdfStruct, const glm::vec3& wo, glm::vec3& wi, float* pdf, thrust::default_random_engine & rng, const glm::vec2 & uv, const glm::vec3 & rands) {
    switch (bsdfStruct.bsdfType)
    {
    case DIFFUSE:
    {
        wi = hemiSphereRandomSample(glm::vec2(rands), pdf);
        // Encapsulate a sampler class
        return f(bsdfStruct, wo, wi, uv);
    }
    case MICROFACET:
        // Uniform sampling for now
        // TODO: Add importance sampling

        //wi = hemiSphereRandomSample(rng, pdf);
        //return f(bsdfStruct, wo, wi, uv);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float alpha = bsdfStruct.roughnessFactor * bsdfStruct.roughnessFactor;
        glm::vec3 wi_diffuse;
        float pdf_diffuse;
        //wi_diffuse = hemiSphereRandomSample(glm::vec2(rands), &pdf_diffuse);
        glm::vec3 wh;
        if (rands.z > (1.f / (2.f - bsdfStruct.metallicFactor))) {
            wi = hemiSphereRandomSample(glm::vec2(rands), &pdf_diffuse);
            wh = glm::normalize(wi + wo);
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
            if (d < 0) return glm::vec3();
            wi = glm::reflect(-wo, wh);
        }
        *pdf = PDF(bsdfStruct, wo, wi, rands);
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