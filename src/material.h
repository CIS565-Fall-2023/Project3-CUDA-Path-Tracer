#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <cuda_runtime.h>

#include "sceneStructs.h"
#include "utilities.h"

__device__ __inline__ glm::vec3 sampleTexture(cudaTextureObject_t tex, glm::vec2 const& uv) {
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(color.x, color.y, color.z);
}

__device__ __inline__ glm::vec2 sampleSphericalMap(glm::vec3 v) {
    glm::vec2 uv = glm::vec2(glm::atan(v.z, v.x), asin(v.y));
    uv *= glm::vec2(0.1591, 0.3183);
    uv += 0.5;
    return uv;
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

    bool entering = cosThetaI > 0.f;
    if (!entering)
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
    float ratio = (Rparl * Rparl + Rperp * Rperp) / 2;
    return ratio;
}

__device__ glm::vec3 fresnelDieletricConductor(float cosThetaI, glm::vec3 etat, glm::vec3 k)
{

    glm::vec3 etai = glm::vec3(1, 1, 1);
#if Au
    glm::vec3 etat = glm::vec3(0.142820060, 0.374143630, 1.43944442);
    glm::vec3 k = glm::vec3(3.97471833, 2.38065982, 1.59981036);
#else
#if Cu
    glm::vec3 etat = glm::vec3(0.199023, 0.924777, 1.09564);
    glm::vec3 k = glm::vec3(3.89346, 2.4567, 2.13024);
#else
#if Al
    glm::vec3 etat = glm::vec3(1.64884, 0.881674, 0.518685);
    glm::vec3 k = glm::vec3(9.18441, 6.27709, 4.81076);
#endif
#endif
#endif
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

__device__ glm::vec3 sample_f_metal(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, Material::Metal metal, glm::vec3 wo, BsdfSample& sample)
{
    float random = xi.z;
    if (random < 0.5)
    {
        glm::vec3 R = sample_f_specular_refl(albedo, nor, wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * fresnelDieletricConductor(dot(nor, normalize(sample.wiW)), metal.etat, metal.k) * R;
    }
    else
    {
        glm::vec3 T = sample_f_specular_trans(albedo, nor, glm::length(metal.etat), wo, sample);
        sample.sampledType = BsdfSampleType::SPEC_REFL | BsdfSampleType::SPEC_TRANS;
        return 2.f * (glm::vec3(1.) - fresnelDieletricConductor(dot(nor, normalize(sample.wiW)), metal.etat, metal.k)) * T;
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
    float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
    cosTheta = 1 / glm::sqrt(1 + tanTheta2);

    float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    if (!SameHemisphere(wo, wh))
        wh = -wh;

    return wh;
}

__device__ float trowbridgeReitzD(glm::vec3 wh, float roughness)
{
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta))
        return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e = (Cos2Phi(wh) / (roughness * roughness) +
        Sin2Phi(wh) / (roughness * roughness)) *
        tan2Theta;
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

__device__ float lambda(glm::vec3 w, float roughness)
{
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta))
        return 0.f;

    float alpha = glm::sqrt(Cos2Phi(w) * roughness * roughness +
        Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta =
        (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + glm::sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__device__ float trowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness)
{
    return 1 / (1 + lambda(wo, roughness) + lambda(wi, roughness));
}

__device__ float trowbridgeReitzPdf(glm::vec3 wo, glm::vec3 wh, float roughness)
{
    return trowbridgeReitzD(wh, roughness) * AbsCosTheta(wh);
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

__device__ glm::vec3 sample_f_microfacet_refl(glm::vec3 albedo, glm::vec3 nor, glm::vec3 xi, glm::vec3 wo,

    float roughness, BsdfSample& sample)
{
    if (wo.z == 0)
        return glm::vec3(0.f);
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    glm::vec3 wi = reflect(-wo, wh);
    sample.wiW = LocalToWorld(nor) * wi;
    if (!SameHemisphere(wo, wi))
        return glm::vec3(0.f);

    sample.pdf = trowbridgeReitzPdf(wo, wh, roughness) / (4 * dot(wo, wh));
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

__device__ glm::vec3 computeAlbedo(const Material& mat, glm::vec3 nor, glm::vec2 uv)
{
    glm::vec3 albedo(1.f);
    if ((mat.type & (Material::Type::PLASTIC | Material::Type::DIFFUSE)) != 0)
    {
        auto& tex = mat.pbrMetallicRoughness.baseColorTexture;
        if (tex.index != -1)
            albedo = sampleTexture(tex.cudaTexObj, uv);
        else
            albedo = glm::vec3(mat.pbrMetallicRoughness.baseColorFactor);
    }
    else if (mat.type == Material::Type::UNKNOWN) {
        albedo = glm::vec3(0.f);
    }
#if N_TEXTURES
    if (mat.albedoTex != -1)
    {
        albedo *=
            texture(u_Texsamplers[mat.albedoTex], isect.uv).rgb;
        albedo.rgb = pow(albedo, glm::vec3(2.2));
    }
#endif
    return albedo;
}

__device__ float computeRoughness(const Material& mat, glm::vec3 nor)
{
    float roughness = 0.f;
    if (mat.type == Material::Type::PLASTIC)
        roughness = mat.pbrMetallicRoughness.roughnessFactor;
#if N_TEXTURES
    if (mat.roughnessTex != -1)
    {
        roughness =
            texture(u_Texsamplers[mat.roughnessTex], isect.uv).r;
    }
#endif
    return roughness;
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

__device__ glm::vec3 sample_f(const Material& mat, glm::vec3 nor, glm::vec2 uv, glm::vec3 woW, glm::vec3 xi, BsdfSample& sample)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;
    if (mat.type == Material::Type::ROUGH_DIELECTRIC)
    {
        return sample_f_rough_dieletric(computeAlbedo(mat, nor, uv), nor, xi, wo, computeRoughness(mat, nor), mat.dielectric.eta, sample);
    }
    else if (mat.type == Material::Type::SPECULAR)
    {
        sample.pdf = 1.;
        return sample_f_specular_refl(computeAlbedo(mat, nor, uv), nor, wo, sample);
    }
    else if (mat.type == Material::Type::METAL)
    {
        sample.pdf = 1.;
        return sample_f_metal(computeAlbedo(mat, nor, uv), nor, xi, mat.metal, wo, sample);
    }
    else if (mat.type == Material::Type::DIELECTRIC)
    {
        sample.pdf = 1.;
        return sample_f_glass(computeAlbedo(mat, nor, uv), nor, xi, mat.dielectric.eta, wo, sample);
    }
    else if (mat.type == Material::Type::DIFFUSE)
    {
        return sample_f_diffuse(computeAlbedo(mat, nor, uv), xi, nor, sample);
    }
    sample.pdf = -1.f;
    return glm::vec3(0.f);
}

__device__ float pdf(const Material& mat, glm::vec3 nor, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 wi = WorldToLocal(nor) * wiW;

    /*if (wo.z == 0)

        if (mat.type == BsdfSampleType::DIFFUSE_REFL)
        {
            return squareToHemisphereCosinePDF(wi);
        }
        else if (mat.type == BsdfSampleType::SPEC_REFL ||
            mat.type == BsdfSampleType::SPEC_TRANS ||
            mat.type == BsdfSampleType::SPEC_GLASS)
        {
            return 0.f;
        }
        else if (mat.type == BsdfSampleType::MICROFACET_REFL)
        {
            glm::vec3 wh = normalize(wo + wi);
            return trowbridgeReitzPdf(wo, wh, computeRoughness(mat, nor)) /
                (4 * dot(wo, wh));
        }
        else if (mat.type == BsdfSampleType::MICROFACET_TRANS)
        {
            return pdf_microfacet_trans(wo, wi, mat.eta, computeRoughness(mat, nor));
        }
        else
        {
            return 0.f;
        }*/
    return 0.f;
}
