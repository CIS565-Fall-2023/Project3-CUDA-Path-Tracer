#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

struct BsdfSample
{
    glm::vec3 f;
    glm::vec3 wiW;
    float pdf;
    int sampledType;
};

enum BsdfSampleType
{
    diffuse_refl = 1,
    spec_refl = 1 << 1,
    spec_trans = 1 << 2,
    spec_glass = 1 << 3,
    microfacet_refl = 1 << 4,
    plastic = 1 << 5,
    diffuse_trans = 1 << 6,
    microfacet_trans = 1 << 7;
};

__device__ glm::vec3 f_diffuse(glm::vec3 albedo)
{
    return albedo / PI;
}

__device__ glm::vec3 sample_f_diffuse(glm::vec3 albedo, glm::vec2 xi, glm::vec3 nor, glm::vec3& wiW,
    float& pdf, int& sampledType)
{
    glm::vec3 wi = squareToHemisphereCosine(xi);
    pdf = squareToHemisphereCosinePDF(wi);
    wiW = LocalToWorld(nor) * wi;
    sampledType = DIFFUSE_REFL;
    return f_diffuse(albedo);
}

__device__ glm::vec3 sample_f_specular_refl(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo, glm::vec3& wiW,
    int& sampledType)
{
    glm::vec3 wi = reflect(-wo, glm::vec3(0, 0, 1));
    wiW = LocalToWorld(nor) * wi;
    sampledType = SPEC_REFL;
    if (wi.z == 0)
        return glm::vec3(0.);
    return albedo / AbsCosTheta(wi);
}

__device__ glm::vec3 sample_f_specular_trans(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo, glm::vec3& wiW,
    int& sampledType)
{
    float etaA = 1.;
    float etaB = 1.55;
    bool entering = CosTheta(wo) > 0;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    glm::vec3 wi;
    if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi) || wi.z == 0)
    {
        return glm::vec3(0.);
    }
    wiW = LocalToWorld(nor) * wi;
    sampledType = SPEC_TRANS;
    return albedo / AbsCosTheta(wi);
}

__device__ glm::vec3 fresnelDielectricEval(float cosThetaI, float eta)
{
    float etaI = 1.;
    float etaT = eta;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.f;
    if (!entering)
    {
        float etaT = 1.;
        float etaI = 1.55;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.)
        return glm::vec3(1.);
    float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    float ratio = (Rparl * Rparl + Rperp * Rperp) / 2;
    return glm::vec3(ratio);
}

__device__ glm::vec3 fresnelDieletricConductor(float cosThetaI)
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
#else
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

__device__ glm::vec3 sample_f_metal(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo, glm::vec3& wiW,
    int& sampledType)
{
    float random = rng();
    if (random < 0.5)
    {
        glm::vec3 R = sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_REFL;
        return 2.f * fresnelDieletricConductor(dot(nor, normalize(wiW))) * R;
    }
    else
    {
        glm::vec3 T = sample_f_specular_trans(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_TRANS;
        return 2.f *
            (glm::vec3(1.) -
                fresnelDieletricConductor(dot(nor, normalize(wiW)))) *
            T;
    }
}

__device__ glm::vec3 sample_f_glass(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo, glm::vec3& wiW,
    int& sampledType)
{
    float random = rng();
    if (random < 0.5)
    {
        glm::vec3 R = sample_f_specular_refl(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_REFL;
        return 2.f * fresnelDielectricEval(dot(nor, normalize(wiW)), 1.55) * R;
    }
    else
    {
        glm::vec3 T = sample_f_specular_trans(albedo, nor, wo, wiW, sampledType);
        sampledType = SPEC_TRANS;
        return 2.f *
            (glm::vec3(1.) -
                fresnelDielectricEval(dot(nor, normalize(wiW)), 1.55)) *
            T;
    }
}

__device__ glm::vec3 sample_wh(glm::vec3 wo, glm::vec2 xi, float roughness)
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

__device__ glm::vec3 sample_f_microfacet_refl(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo,

    float roughness, glm::vec3& wiW, float& pdf,
    int& sampledType)
{
    if (wo.z == 0)
        return glm::vec3(0.f);
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    glm::vec3 wi = reflect(-wo, wh);
    wiW = LocalToWorld(nor) * wi;
    if (!SameHemisphere(wo, wi))
        return glm::vec3(0.f);

    pdf = trowbridgeReitzPdf(wo, wh, roughness) / (4 * dot(wo, wh));
    return f_microfacet_refl(albedo, wo, wi, roughness);
}

__device__ glm::vec3 sample_f_microfacet_trans(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo,
    float roughness, float etaB, glm::vec3& wiW,
    float& pdf, int& sampledType)
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
    wiW = LocalToWorld(nor) * wi;
    pdf = pdf_microfacet_trans(wo, wi, etaB, roughness);
    return f_microfacet_trans(albedo, wo, wi, etaB, roughness);
}

__device__ glm::vec3 sample_f_rough_dieletric(glm::vec3 albedo, glm::vec3 nor, glm::vec2 xi, glm::vec3 wo,
    float roughness, float eta, glm::vec3& wiW,
    float& pdf, int& sampledType)
{
    glm::vec3 wh = sample_wh(wo, xi, roughness);
    float pr = fresnelDielectricEval(dot(wo, wh), eta).x;
    float pt = 1 - pr;
    float random = rng();
    if (random < pr)
    {
        glm::vec3 R = sample_f_microfacet_refl(albedo, nor, xi, wo, roughness, wiW,
            pdf, sampledType);
        sampledType = MICROFACET_REFL;
        pdf *= pr;
        return R * pr;
    }
    else
    {
        glm::vec3 T = sample_f_microfacet_trans(albedo, nor, xi, wo, roughness, eta,
            wiW, pdf, sampledType);
        sampledType = MICROFACET_TRANS;
        pdf *= pt;
        return T * pt;
    }
}

__device__ glm::vec3 computeAlbedo(const Material mat, glm::vec3 nor)
{
    glm::vec3 albedo = isect.material.albedo;
#if N_TEXTURES
    if (isect.material.albedoTex != -1)
    {
        albedo *=
            texture(u_Texsamplers[isect.material.albedoTex], isect.uv).rgb;
        albedo.rgb = pow(albedo, glm::vec3(2.2));
    }
#endif
    return albedo;
}

__device__ glm::vec3 computeNormal(const Material mat, glm::vec3 nor)
{
    glm::vec3 nor = isect.nor;
#if N_TEXTURES
    if (isect.material.normalTex != -1)
    {
        glm::vec3 localNor =
            texture(u_Texsamplers[isect.material.normalTex], isect.uv).rgb;
        glm::vec3 tan, bit;
        coordinateSystem(nor, tan, bit);
        nor = mat3(tan, bit, nor) * localNor;
    }
#endif
    return nor;
}

__device__ float computeRoughness(const Material mat, glm::vec3 nor)
{
    float roughness = isect.material.roughness;
#if N_TEXTURES
    if (isect.material.roughnessTex != -1)
    {
        roughness =
            texture(u_Texsamplers[isect.material.roughnessTex], isect.uv).r;
    }
#endif
    return roughness;
}

__device__ glm::vec3 f(const Material mat, glm::vec3 nor, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 nor = computeNormal(isect);
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0)
        return glm::vec3(0.f);

    if (isect.material.type == DIFFUSE_REFL)
    {
        return f_diffuse(computeAlbedo(isect));
    }
    else if (isect.material.type == SPEC_REFL ||
        isect.material.type == SPEC_TRANS ||
        isect.material.type == SPEC_GLASS)
    {
        return glm::vec3(0.f);
    }
    else if (isect.material.type == MICROFACET_REFL)
    {
        return f_microfacet_refl(computeAlbedo(isect), wo, wi,
            computeRoughness(isect));
    }
    else if (isect.material.type == MICROFACET_TRANS)
    {
        return f_microfacet_trans(computeAlbedo(isect), wo, wi,
            isect.material.eta,
            computeRoughness(isect));
    }
    else
    {
        return glm::vec3(1, 0, 1);
    }
}

__device__ glm::vec3 sample_f(const Material mat, glm::vec3 nor, glm::vec3 woW, glm::vec2 xi, BsdfSample& sample)
{
    glm::vec3 wo = WorldToLocal(nor) * woW;

    if (isect.material.type == DIFFUSE_REFL)
    {
        return sample_f_diffuse(computeAlbedo(isect), xi, nor, wiW, pdf,
            sampledType);
    }
    else if (isect.material.type == SPEC_REFL)
    {
        pdf = 1.;
        return sample_f_specular_refl(computeAlbedo(isect), nor, wo, wiW,
            sampledType);
    }
    else if (isect.material.type == SPEC_TRANS)
    {
        pdf = 1.;
        return sample_f_specular_trans(computeAlbedo(isect), nor, wo, wiW,
            sampledType);
    }
    else if (isect.material.type == SPEC_GLASS)
    {
        pdf = 1.;
#if Al || Au || Cu
        return sample_f_metal(computeAlbedo(isect), nor, xi, wo, wiW,
            sampledType);
#else
        return sample_f_glass(computeAlbedo(isect), nor, xi, wo, wiW,
            sampledType);
#endif
    }
    else if (isect.material.type == MICROFACET_REFL)
    {
        return sample_f_microfacet_refl(computeAlbedo(isect), nor, xi, wo,
            computeRoughness(isect), wiW, pdf,
            sampledType);
    }
    else if (isect.material.type == MICROFACET_TRANS)
    {
        return sample_f_rough_dieletric(
            computeAlbedo(isect), nor, xi, wo, computeRoughness(isect),
            isect.material.eta, wiW, pdf, sampledType);
    }
    else if (isect.material.type == PLASTIC)
    {
        return glm::vec3(1, 0, 1);
    }
    else
    {
        return glm::vec3(1, 0, 1);
    }
}

__device__ float pdf(const Material mat, glm::vec3 nor, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 nor = computeNormal(isect);
    glm::vec3 wo = WorldToLocal(nor) * woW;
    glm::vec3 wi = WorldToLocal(nor) * wiW;

    if (wo.z == 0)

        if (isect.material.type == DIFFUSE_REFL)
        {
            return squareToHemisphereCosinePDF(wi);
        }
        else if (isect.material.type == SPEC_REFL ||
            isect.material.type == SPEC_TRANS ||
            isect.material.type == SPEC_GLASS)
        {
            return 0.f;
        }
        else if (isect.material.type == MICROFACET_REFL)
        {
            glm::vec3 wh = normalize(wo + wi);
            return trowbridgeReitzPdf(wo, wh, computeRoughness(isect)) /
                (4 * dot(wo, wh));
        }
        else if (isect.material.type == MICROFACET_TRANS)
        {
            return pdf_microfacet_trans(wo, wi, isect.material.eta, computeRoughness(isect));
        }
        else
        {
            return 0.f;
        }
}
