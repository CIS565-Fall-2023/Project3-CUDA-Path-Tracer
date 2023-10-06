#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <cuda_runtime.h>

#include "sceneStructs.h"
#include "utilities.h"

__host__ __device__ glm::vec3 checkerBoard(const glm::vec2 uv, float scale = 8.f) {
    glm::vec2 scaled_uv = scale * uv;
    return (int(scaled_uv.x) + int(scaled_uv.y)) % 2 == 0 ? glm::vec3(0.f) : glm::vec3(1.f);
}

__device__ __inline__ glm::vec3 sampleTexture(cudaTextureObject_t tex, glm::vec2 const& uv) {
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(color.x, color.y, color.z);
}

__device__ __inline__ glm::vec3 sampleColorTexture(cudaTextureObject_t tex, glm::vec2 const& uv, bool isProcedural, float scale) {
    if (isProcedural)
        return checkerBoard(uv, scale);
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(color.x, color.y, color.z);
}

__device__ __inline__ glm::vec3 sampleEnvTexture(cudaTextureObject_t tex, glm::vec2 const& uv) {
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return glm::clamp(glm::vec3(color.x, color.y, color.z), 0.f, 1.f);
}

__device__ __inline__ float2 texMetallicRoughness(cudaTextureObject_t tex, glm::vec2 const& uv) {
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return float2{ color.z, color.y };
}

__device__ __inline__ glm::vec2 sampleSphericalMap(glm::vec3 v) {
    glm::vec2 uv = glm::vec2(glm::atan(v.z, v.x), asin(v.y));
    uv *= glm::vec2(0.1591, 0.3183);
    uv += 0.5;
    return uv;
}

__device__ __inline__ float colorToGreyscale(glm::vec3 col) {
    return 0.299f * col.r + 0.587f * col.g + 0.114f * col.b;
}

struct BsdfSample
{
    glm::vec3 wiW;
    float pdf;
    uint32_t sampledType;
};
__device__ glm::vec3 f_diffuse(glm::vec3 albedo)
{
    return albedo / PI;
}

__device__ glm::vec3 sample_f_diffuse(glm::vec3 albedo, glm::vec3 xi, glm::vec3 nor, BsdfSample& sample)
{
    glm::vec3 wi = squareToHemisphereCosine(glm::vec2(xi.x, xi.y));
    sample.pdf = squareToHemisphereCosinePDF(wi);
    sample.wiW = LocalToWorld(nor) * wi;
    sample.sampledType = BsdfSampleType::DIFFUSE_REFL;
    return f_diffuse(albedo);
}

__device__ glm::vec3 sample_f_specular_refl(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo, BsdfSample& sample)
{
    glm::vec3 wi = reflect(-wo, glm::vec3(0, 0, 1));
    sample.wiW = LocalToWorld(nor) * wi;
    sample.sampledType = BsdfSampleType::SPEC_REFL;
    if (wi.z == 0)
        return glm::vec3(0.);
    return albedo / AbsCosTheta(wi);
}

__device__ glm::vec3 sample_f_specular_trans(glm::vec3 albedo, glm::vec3 nor, float eta, glm::vec3 wo, BsdfSample& sample)
{
    float etaA = 1.;
    float etaB = eta;
    bool entering = CosTheta(wo) > 0;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    glm::vec3 wi;
    if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi) || wi.z == 0)
    {
        return glm::vec3(0.);
    }
    sample.wiW = LocalToWorld(nor) * wi;
    sample.sampledType = BsdfSampleType::SPEC_TRANS;
    return albedo / AbsCosTheta(wi);
}

__device__ float fresnelDielectricEval(float cosThetaI, float eta)
{
    float etaI = 1.;
    float etaT = eta;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    if (cosThetaI < 0.f)
    {
        etaT = 1.;
        etaI = eta;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.)
        return 1.f;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__device__ inline glm::vec3 fresnelSchlick(float cosThetaI, glm::vec3 f0) {
    return glm::mix(f0, glm::vec3(1.f), powf(1.f - cosThetaI, 5.f));
}

__device__ glm::vec3 fresnelDieletricConductor(float cosThetaI, glm::vec3 etat, glm::vec3 k)
{

    glm::vec3 etai = glm::vec3(1, 1, 1);
    cosThetaI = glm::clamp(abs(cosThetaI), -1.f, 1.f);
    glm::vec3 eta = etat / etai;
    glm::vec3 etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1. - cosThetaI2;
    glm::vec3 eta2 = eta * eta;
    glm::vec3 etak2 = etak * etak;

    glm::vec3 t0 = eta2 - etak2 - sinThetaI2;
    glm::vec3 a2plusb2 = glm::sqrt(t0 * t0 + 4.f * eta2 * etak2);
    glm::vec3 t1 = a2plusb2 + cosThetaI2;
    glm::vec3 a = glm::sqrt(0.5f * (a2plusb2 + t0));
    glm::vec3 t2 = 2.f * cosThetaI * a;
    glm::vec3 Rs = (t1 - t2) / (t1 + t2);

    glm::vec3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    glm::vec3 t4 = t2 * sinThetaI2;
    glm::vec3 Rp = Rs * (t3 - t4) / (t3 + t4);
    glm::vec3 ratio = 0.5f * (Rp + Rs);
    return ratio;
}

__device__ glm::vec3 sample_f_metal(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, const Material::Metal& metal, glm::vec3 wo, BsdfSample& sample)
{
    float random = xi.z;
    const glm::vec3& etat = metal.etat;
    const glm::vec3& k = metal.k;
    if (random < 0.5)
    {
        glm::vec3 R = sample_f_specular_refl(albedo, nor, wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * fresnelDieletricConductor(dot(nor, normalize(sample.wiW)), etat, k) * R;
    }
    else
    {
        glm::vec3 T = sample_f_specular_trans(albedo, nor, glm::length(etat), wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * (glm::vec3(1.) - fresnelDieletricConductor(dot(nor, normalize(sample.wiW)), etat, k)) * T;
    }
}

__device__ glm::vec3 sample_f_glass(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, float indexOfRefraction, glm::vec3 wo, BsdfSample& sample)
{
    float random = xi.z;
    if (random < 0.5)
    {
        glm::vec3 R = sample_f_specular_refl(albedo, nor, wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * fresnelDielectricEval(dot(nor, normalize(sample.wiW)), indexOfRefraction) * R;
    }
    else
    {
        glm::vec3 T = sample_f_specular_trans(albedo, nor, indexOfRefraction, wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * (glm::vec3(1.) - fresnelDielectricEval(dot(nor, normalize(sample.wiW)), indexOfRefraction)) * T;
    }
}

__device__ glm::vec3 sample_wh(glm::vec3 wo, glm::vec3 xi, float roughness)
{
    glm::vec3 wh;

    float cosTheta = 0;
    float phi = TWO_PI * xi[1];
    float tanTheta2 = roughness * roughness * xi[0] / (1.f - xi[0]);
    cosTheta = 1 / glm::sqrt(1 + tanTheta2);

    float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh))
        wh = -wh;

    return wh;
}

// http://graphics.stanford.edu/courses/cs348b-18-spring-content/lectures/12_reflection2/12_reflection2_slides.pdf
__device__ float trowbridgeReitzD(glm::vec3 wh, float roughness)
{
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta))
        return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    if (cos4Theta < 1e-16f)
        return 0;
    float e = tan2Theta * (Sqr(CosPhi(wh) / roughness) + Sqr(SinPhi(wh) / roughness));
    return 1 / (PI * roughness * roughness * cos4Theta * Sqr(1 + e));
}

__device__ float lambda(glm::vec3 w, float roughness)
{
    float tan2Theta = Tan2Theta(w);
    if (isinf(tan2Theta))
        return 0.f;

    float alpha2 = Sqr(Cos2Phi(w) * roughness) + Sqr(Sin2Phi(w) * roughness);
    return (glm::sqrt(1.f + alpha2 * tan2Theta) - 1) / 2.f;
}

__device__ float trowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness)
{
    return 1 / (1 + lambda(wo, roughness) + lambda(wi, roughness));
}

__device__ float trowbridgeReitzG1(glm::vec3 wo, float roughness)
{
    return 1 / (1 + lambda(wo, roughness));
}

__device__ float trowbridgeReitzPdf(glm::vec3 wo, glm::vec3 wh, float roughness)
{
    return trowbridgeReitzG1(wo, roughness) / AbsCosTheta(wo) * trowbridgeReitzD(wh, roughness) * AbsDot(wo, wh);
}

__device__ float pdf_microfacet_trans(glm::vec3 wo, glm::vec3 wi, float etaB, float roughness)
{
    if (SameHemisphere(wo, wi))
        return 0;
    float eta = CosTheta(wo) > 0 ? (etaB / 1.) : (1. / etaB);
    glm::vec3 wh = normalize(wo + wi * eta);

    if (dot(wo, wh) * dot(wi, wh) > 0)
        return 0;
    float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
    float dwh_dwi = abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return trowbridgeReitzPdf(wo, wh, roughness) * dwh_dwi;
}

__device__ glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness)
{
    float cosThetaO = AbsCosTheta(wo);
    float cosThetaI = AbsCosTheta(wi);
    glm::vec3 wh = wi + wo;
    if (cosThetaI == 0 || cosThetaO == 0)
        return glm::vec3(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0)
        return glm::vec3(0.f);
    wh = normalize(wh);
    glm::vec3 F(1.); // fresnel->Evaluate(glm::dot(wi, wh));
    float D = trowbridgeReitzD(wh, roughness);
    float G = trowbridgeReitzG(wo, wi, roughness);
    return albedo * D * G * F / (4 * cosThetaI * cosThetaO);
}

__device__ glm::vec3 f_microfacet_trans(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float etaB, float roughness)
{
    if (SameHemisphere(wo, wi))
        return glm::vec3(0);

    float cosThetaO = CosTheta(wo);
    float cosThetaI = CosTheta(wi);
    if (cosThetaI == 0 || cosThetaO == 0)
        return glm::vec3(0);

    float eta = CosTheta(wo) > 0 ? (etaB / 1.) : (1. / etaB);
    glm::vec3 wh = normalize(wo + wi * eta);
    if (wh.z < 0)
        wh = -wh;

    if (dot(wo, wh) * dot(wi, wh) > 0)
        return glm::vec3(0);

    float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);

    return albedo * abs(trowbridgeReitzD(wh, roughness) * trowbridgeReitzG(wo, wi, roughness) * eta * eta *
        AbsDot(wi, wh) * AbsDot(wo, wh) /
        (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

__device__ glm::vec3 sample_f_microfacet_refl(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, glm::vec3 wo, float roughness, BsdfSample& sample)
{
    if (wo.z == 0)
        return glm::vec3(0.f);
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    glm::vec3 wi = reflect(-wo, wh);
    sample.wiW = LocalToWorld(nor) * wi;
    if (!SameHemisphere(wo, wi))
        return glm::vec3(0.f);

    sample.pdf = trowbridgeReitzPdf(wo, wh, roughness) / (4 * AbsDot(wo, wh));
    return f_microfacet_refl(albedo, wo, wi, roughness);
}

__device__ glm::vec3 sample_f_microfacet_trans(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, glm::vec3 wo,
    float roughness, float etaB, BsdfSample& sample)
{
    float etaA = 1.;
    if (wo.z == 0)
        return glm::vec3(0.f);
    glm::vec3 wi;
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    if (dot(wo, wh) < 0)
        return glm::vec3(0.f);

    float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
    if (!Refract(wo, wh, eta, wi))
        return glm::vec3(0);
    sample.wiW = LocalToWorld(nor) * wi;
    sample.pdf = pdf_microfacet_trans(wo, wi, etaB, roughness);
    return f_microfacet_trans(albedo, wo, wi, etaB, roughness);
}

__device__ glm::vec3 sample_f_rough_dieletric(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, glm::vec3 wo,
    float roughness, float eta, BsdfSample& sample)
{
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    float pr = fresnelDielectricEval(dot(wo, wh), eta);
    float pt = 1 - pr;
    float random = xi.z;
    if (random < pr)
    {
        glm::vec3 R = sample_f_microfacet_refl(albedo, nor, xi, wo, roughness, sample);
        sample.sampledType = BsdfSampleType::MICROFACET_REFL;
        sample.pdf *= pr;
        return R * pr;
    }
    else
    {
        glm::vec3 T = sample_f_microfacet_trans(albedo, nor, xi, wo, roughness, eta, sample);
        sample.sampledType = BsdfSampleType::MICROFACET_TRANS;
        sample.pdf *= pt;
        return T * pt;
    }
}

__device__ glm::vec3 sample_f_pbrMetRough(glm::vec3 albedo, const float2& metallicRoughness, const Material& material, glm::vec3 nor, glm::vec3 xi, glm::vec3 wo, BsdfSample& sample)
{
    float random = xi.z;
    float metallic = metallicRoughness.x, roughness = metallicRoughness.y;
    const PbrMetallicRoughness& pbrMaterial = material.pbrMetallicRoughness;
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    float ior = material.dielectric.eta;
    glm::vec3 fr = fresnelSchlick(AbsDot(wo, wh), glm::min(pow(((1 - ior) / (1 + ior)), 2.f) * material.specular.specularColorFactor, glm::vec3(1.f)));
    glm::vec3 weightSpec = material.specular.specularFactor * fr;
    float pSpec = colorToGreyscale(weightSpec) * (1 - metallic);
    float pDiff = (1.f - material.specular.specularFactor * glm::min(glm::min(fr.x, fr.y), fr.z)) * (1 - metallic);
    float pMetal = metallic;
    float total = pSpec + pDiff + pMetal;
    pSpec /= total;
    pDiff /= total;
    pMetal /= total;

    if (random < pSpec) {
        glm::vec3 R;
        if (roughness == 0.f) {
            sample.pdf = 1.f;
            R = sample_f_specular_refl(albedo, nor, wo, sample);
        }
        else
            R = sample_f_microfacet_refl(albedo, nor, xi, wo, roughness * roughness, sample);
        return R;
    }
    else if (random < pSpec + pMetal) {
        glm::vec3 wh = sample_wh(wo, xi, roughness);
        glm::vec3 R;
        if (roughness == 0.f) {
            sample.pdf = 1.f;
            R = sample_f_specular_refl(albedo, nor, wo, sample);
        }
        else
            R = sample_f_microfacet_refl(albedo, nor, xi, wo, roughness * roughness, sample);
        return R;
    }
    else {
        glm::vec3 D = sample_f_diffuse(albedo, xi, nor, sample);
        return D;
    }
}

__device__ float2 getMetallic(const Material& mat, glm::vec2 uv) {
    float2 metallicRoughness;
    auto& tex = mat.pbrMetallicRoughness.metallicRoughnessTexture;
    if (tex.index != -1)
        metallicRoughness = texMetallicRoughness(tex.cudaTexObj, uv);
    else
        metallicRoughness = float2{ (float)mat.pbrMetallicRoughness.metallicFactor,(float)mat.pbrMetallicRoughness.roughnessFactor };
    return metallicRoughness;
}

__device__ glm::vec3 computeAlbedo(const Material& mat, glm::vec3 nor, glm::vec2 uv, bool isProcedural, float scale)
{
    glm::vec3 albedo(1.f);
    if ((mat.type == Material::Type::DIFFUSE || (mat.type == Material::Type::PBR)))
    {
        auto& tex = mat.pbrMetallicRoughness.baseColorTexture;
        if (tex.index != -1)
            albedo = sampleColorTexture(tex.cudaTexObj, uv, isProcedural, scale);
        else
            albedo = glm::vec3(mat.pbrMetallicRoughness.baseColorFactor);
    }
    else if (mat.type == Material::Type::METAL) {
        albedo = glm::vec3(1.f);
    }
    else if (mat.type == Material::Type::UNKNOWN) {
        albedo = glm::vec3(0.f);
    }
    return albedo;
}


__device__ glm::vec3 f(const Material& mat, glm::vec3 nor, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0)
        return glm::vec3(0.f);

    /*if (mat.type == BsdfSampleType::DIFFUSE_REFL)
    {
        return f_diffuse(computeAlbedo(mat, nor));
    }
    else if (mat.type == BsdfSampleType::SPEC_REFL ||
        mat.type == BsdfSampleType::SPEC_TRANS ||
        mat.type == BsdfSampleType::SPEC_GLASS)
    {
        return glm::vec3(0.f);
    }
    else if (mat.type == BsdfSampleType::MICROFACET_REFL)
    {
        return f_microfacet_refl(computeAlbedo(mat, nor), wo, wi,
            computeRoughness(mat, nor));
    }
    else if (mat.type == BsdfSampleType::MICROFACET_TRANS)
    {
        return f_microfacet_trans(computeAlbedo(mat, nor), wo, wi,
            mat.eta,
            computeRoughness(mat, nor));
    }
    else
    {
        return glm::vec3(1, 0, 1);
    }*/
    return glm::vec3(1, 0, 1);
}

__device__ glm::vec3 sample_f(const Material& mat, bool isProcedural, float scale, glm::vec3 nor, glm::vec2 uv, glm::vec3 woW, glm::vec3 xi, BsdfSample& sample)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 albedo = computeAlbedo(mat, nor, uv, isProcedural, scale);
    if (mat.type == Material::Type::ROUGH_DIELECTRIC)
    {
        return sample_f_rough_dieletric(albedo, nor, xi, wo, getMetallic(mat, uv).y, mat.dielectric.eta, sample);
    }
    else if (mat.type == Material::Type::SPECULAR)
    {
        sample.pdf = 1.;
        return sample_f_specular_refl(albedo, nor, wo, sample);
    }
    else if (mat.type == Material::Type::METAL)
    {
        sample.pdf = 1.;
        return sample_f_metal(albedo, nor, xi, mat.metal, wo, sample);
    }
    else if (mat.type == Material::Type::DIELECTRIC)
    {
        sample.pdf = 1.;
        return sample_f_glass(albedo, nor, xi, mat.dielectric.eta, wo, sample);
    }
    else if (mat.type == Material::Type::DIFFUSE)
    {
        return sample_f_diffuse(albedo, xi, nor, sample);
    }
    else if (mat.type == Material::Type::PBR)
    {
        return sample_f_pbrMetRough(albedo, getMetallic(mat, uv), mat, nor, xi, wo, sample);
    }
    sample.pdf = -1.f;
    return glm::vec3(0.f);
}

__device__ float pdf(const Material& mat, glm::vec3 nor, glm::vec2 uv, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0)

        if (mat.type == BsdfSampleType::DIFFUSE_REFL)
        {
            return squareToHemisphereCosinePDF(wi);
        }
        else if (mat.type == BsdfSampleType::SPEC_REFL ||
            mat.type == BsdfSampleType::SPEC_TRANS)
        {
            return 0.f;
        }
        else if (mat.type == BsdfSampleType::MICROFACET_REFL)
        {
            glm::vec3 wh = normalize(wo + wi);
            return trowbridgeReitzPdf(wo, wh, getMetallic(mat, uv).y) /
                (4 * dot(wo, wh));
        }
        else if (mat.type == BsdfSampleType::MICROFACET_TRANS)
        {
            return pdf_microfacet_trans(wo, wi, mat.dielectric.eta, getMetallic(mat, uv).y);
        }
        else
        {
            return 0.f;
        }
    return 0.f;
}
