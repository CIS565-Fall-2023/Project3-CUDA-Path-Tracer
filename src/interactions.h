#pragma once

#include "intersections.h"

//http://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html
//Get tangent space vectors
__device__ inline void util_math_get_TBN_pixar(const glm::vec3& N, glm::vec3* T, glm::vec3* B)
{
	float x = N.x, y = N.y, z = N.z;
	float sz = z < 0 ? -1 : 1;
	float a = 1.0f / (sz + z);
	float ya = y * a;
	float b = x * ya;
	float c = x * sz;
	(*T) = glm::vec3(c * x * a - 1, sz * b, c);
	(*B) = glm::vec3(b, y * ya - sz, y);
}

__device__ inline void util_math_get_TBN_naive(const glm::vec3& N, const glm::vec3& pos, glm::vec3* T, glm::vec3* B)
{
	//glm::vec3 tmp = abs(N.x) > 1.0 - EPSILON ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
	*T = glm::cross(N, pos);
	*B = glm::cross(N, *T);
}

__device__ inline glm::vec2 util_sample_disk_uniform(const glm::vec2& random)
{
	float r = sqrt(random.x);
	float theta = TWO_PI * random.y;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__device__ inline glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random)
{
	glm::vec2 t = util_sample_disk_uniform(random);
	return glm::vec3(t.x, t.y, sqrt(1 - t.x * t.x - t.y * t.y));
}

__device__ inline float util_math_tangent_space_abscos(const glm::vec3& w)
{
	return abs(w.z);
}

__device__ inline float util_math_sin_cos_convert(float sinOrCos)
{
	return sqrt(max(1 - sinOrCos * sinOrCos, 0.0f));
}

__device__ inline float util_math_frensel_dielectric(float cosThetaI, float etaI, float etaT)
{
	float sinThetaI = util_math_sin_cos_convert(cosThetaI);
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1) return 1;//total reflection
	float cosThetaT = util_math_sin_cos_convert(sinThetaT);
	float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
	return (rparll * rparll + rperpe * rperpe) * 0.5;
}

__device__ inline bool util_geomerty_refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3* wt)
{
	float cosThetaI = glm::dot(wi, n);
	float sin2ThetaI = max(0.0f, 1 - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = sqrt(1 - sin2ThetaT);
	*wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
	return true;
}

__device__ inline float util_math_pow_5(float x)
{
	float x2 = x * x;
	return x2 * x2 * x;
}

__device__ inline glm::vec3 util_math_fschlick(glm::vec3 f0, float HoV)
{
	return f0 + (1.0f - f0) * util_math_pow_5(1.0f - HoV);
}


__device__ inline glm::vec3 util_math_fschlick_roughness(glm::vec3 f0, float roughness, float NoV)
{
	return f0 + util_math_pow_5(1.0f - NoV) * (glm::max(f0, glm::vec3(1.0f - roughness) - f0));
}

__device__ inline float util_math_luminance(glm::vec3 col)
{
	return 0.299f * col.r + 0.587f * col.g + 0.114f * col.b;
}

//https://hal.science/hal-01509746/document
__device__ inline glm::vec3 util_math_sample_ggx_vndf(const glm::vec3& wo, float roughness, const glm::vec2& rand)
{
	glm::vec3 v = glm::normalize(glm::vec3(wo.x * roughness, wo.y * roughness, wo.z));
	glm::vec3 t1 = v.z > 1 - EPSILON ? glm::vec3(1, 0, 0) : glm::cross(v, glm::vec3(0, 0, 1));
	glm::vec3 t2 = glm::cross(t1, v);
	float a = 1 / (1 + v.z);
	float r = sqrt(rand.x);
	float phi = rand.y < a ? rand.y / a * PI : ((rand.y - a) / (1.0 - a) + 1) * PI;
	float p1 = r * cos(phi);
	float p2 = r * sin(phi);
	p2 *= rand.y < a ? 1.0 : v.z;
	glm::vec3 h = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;
	return glm::normalize(glm::vec3(h.x * roughness, h.y * roughness, max(0.0f, h.z)));
}

__device__ inline float util_math_smith_ggx_masking(const glm::vec3& wo, float a2)
{
	float NoV = util_math_tangent_space_abscos(wo);
	return 2 * NoV / (sqrt(NoV * NoV * (1 - a2) + a2) + NoV);
}

__device__ inline float util_math_smith_ggx_shadowing_masking(const glm::vec3& wi, const glm::vec3& wo, float a2)
{
	float NoL = util_math_tangent_space_abscos(wi);
	float NoV = util_math_tangent_space_abscos(wo);
	float denom = NoL * sqrt(NoV * NoV * (1 - a2) + a2) + NoV * sqrt(NoL * NoL * (1 - a2) + a2);
	return (2.0 * NoL * NoV) / (denom);
}

__device__ inline float util_math_ggx_normal_distribution(const glm::vec3& wh, float a2)
{
	float NoH = util_math_tangent_space_abscos(wh);
	float denom = NoH * NoH * (a2 - 1) + 1;
	denom = denom * denom * PI;
	return (a2 + 1e-10) / (denom + 1e-10);
}

__device__ inline glm::vec3 bxdf_diffuse_eval(const glm::vec3& wo, const glm::vec3& wi, glm::vec3& baseColor)
{
	return baseColor * INV_PI;
}

__device__ inline float bxdf_diffuse_pdf(const glm::vec3& wo, const glm::vec3& wi)
{
	return util_math_tangent_space_abscos(wi) * INV_PI;
}

__device__ inline glm::vec3 bxdf_diffuse_sample(const glm::vec2& random, const glm::vec3& wo)
{
	return util_sample_hemisphere_cosine(random);
}

__device__ glm::vec3 bxdf_diffuse_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& random, float* pdf, glm::vec3 diffuseAlbedo)
{
	*wi = util_sample_hemisphere_cosine(random);
	*pdf = util_math_tangent_space_abscos(*wi) * INV_PI;
	return diffuseAlbedo * INV_PI;
}

__device__ glm::vec3 bxdf_frensel_specular_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& random, float* pdf, glm::vec3 reflectionAlbedo, glm::vec3 refractionAlbedo, glm::vec2 refIdx)
{
	float frensel = util_math_frensel_dielectric(util_math_tangent_space_abscos(wo), refIdx.x, refIdx.y);
	if (random.x < frensel)
	{
		*wi = glm::vec3(-wo.x, -wo.y, wo.z);
		*pdf = frensel;
		return frensel * reflectionAlbedo / util_math_tangent_space_abscos(*wi);
	}
	else
	{
		glm::vec3 n = glm::dot(wo, glm::vec3(0, 0, 1)) > 0 ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
		glm::vec3 refractedRay;
		if (!util_geomerty_refract(wo, n, refIdx.x / refIdx.y, &refractedRay)) return glm::vec3(0);
		*wi = refractedRay;
		*pdf = 1 - frensel;
		glm::vec3 val = refractionAlbedo * (1 - frensel) * (refIdx.x * refIdx.x) / (refIdx.y * refIdx.y);
		return val / util_math_tangent_space_abscos(*wi);
	}
}

__device__ inline glm::vec3 bxdf_microfacet_eval(const glm::vec3& wo, const glm::vec3& wi, glm::vec3& baseColor, float roughness)
{
	glm::vec3 wh = glm::normalize(wo + wi);
	float a2 = roughness * roughness;
	glm::vec3 F = util_math_fschlick(baseColor, glm::abs(glm::dot(wo, wh)));
	float D = util_math_ggx_normal_distribution(wh, a2);
	float G2 = util_math_smith_ggx_shadowing_masking(wi, wo, a2);
	return (F * G2 * D) / (4 * util_math_tangent_space_abscos(wo));
}

__device__ inline float bxdf_microfacet_pdf(const glm::vec3& wo, const glm::vec3& wi, float roughness)
{
	float a2 = roughness * roughness;
	glm::vec3 wh = glm::normalize(wo + wi);
	float G1 = util_math_smith_ggx_masking(wo, a2);
	float D = util_math_ggx_normal_distribution(wh, a2);

	return (G1 * D) / (4 * util_math_tangent_space_abscos(wo));
}

__device__ inline glm::vec3 bxdf_microfacet_sample(const glm::vec2& random, const glm::vec3& wo, float roughness)
{
	float a2 = roughness * roughness;
	glm::vec3 h = util_math_sample_ggx_vndf(wo, roughness, random);//importance sample
	glm::vec3 wi = glm::reflect(-wo, h);
	return wi;
}

__device__ inline glm::vec3 bxdf_microfacet_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& random, float* pdf, glm::vec3 reflectionAlbedo, float roughness)
{
	*wi = bxdf_microfacet_sample(random, wo, roughness);
	*pdf = bxdf_microfacet_pdf(wo, *wi, roughness);
	if ((*wi).z > 0)
		return bxdf_microfacet_eval(wo, *wi, reflectionAlbedo, roughness);
	else
		return glm::vec3(0);
}

__device__ inline float util_math_calc_lS(float metallic)
{
	return 1.0 / (2.0 - metallic * metallic);
}

__device__ inline glm::vec3 bxdf_metallic_workflow_eval(const glm::vec3& wo, const glm::vec3& wi, glm::vec3& baseColor, float metallic, float roughness)
{
	glm::vec3 wh = glm::normalize(wo + wi);
	glm::vec3 F0 = glm::vec3(0.04);
	F0 = glm::mix(F0, baseColor, metallic);
	glm::vec3 F = util_math_fschlick(F0, glm::abs(glm::dot(wh, wo)));
	glm::vec3 kS = F;
	glm::vec3 kD = glm::vec3(1.0f) - kS;
	kD *= 1.0 - metallic;
	float a2 = roughness * roughness;
	float D = util_math_ggx_normal_distribution(wh, a2);
	float G2 = util_math_smith_ggx_shadowing_masking(wi, wo, a2);
	return kD * bxdf_diffuse_eval(wo, wi, baseColor) + kS * D * G2 / (4 * util_math_tangent_space_abscos(wo) * util_math_tangent_space_abscos(wi));
}

__device__ inline float bxdf_metallic_workflow_pdf(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& baseColor, float metallic, float roughness)
{
	float lS = util_math_calc_lS(metallic);
	float lD = 1 - lS;
	return lD * bxdf_diffuse_pdf(wo, wi) + lS * bxdf_microfacet_pdf(wo, wi, roughness);
}

__device__ inline glm::vec3 bxdf_metallic_workflow_sample(const glm::vec3& random, const glm::vec3& wo, const glm::vec3& baseColor, float metallic, float roughness)
{
	float lS = util_math_calc_lS(metallic);
	if (random.x < lS)
	{
		return bxdf_microfacet_sample(glm::vec2(random.y, random.z), wo, roughness);
	}
	else
	{
		return bxdf_diffuse_sample(glm::vec2(random.y, random.z), wo);
	}
}

__device__ inline glm::vec3 bxdf_metallic_workflow_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& random, float* pdf, glm::vec3& baseColor, float metallic, float roughness)
{
	*wi = bxdf_metallic_workflow_sample(random, wo, baseColor, metallic, roughness);
	*pdf = bxdf_metallic_workflow_pdf(wo, *wi, baseColor, metallic, roughness);
	return bxdf_metallic_workflow_eval(wo, *wi, baseColor, metallic, roughness);
}


