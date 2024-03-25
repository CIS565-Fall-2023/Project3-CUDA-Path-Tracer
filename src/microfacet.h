#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "utilities.h"
#include "randomUtils.h"
#include "sceneStructs.h"

__device__ float util_alpha_i(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	const float invSinTheta2 = 1.0f / (1.0f - wi.z * wi.z);
	const float cosPhi2 = wi.x * wi.x * invSinTheta2;
	const float sinPhi2 = wi.y * wi.y * invSinTheta2;
	const float alpha_i = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
	return alpha_i;
}

__device__ inline float util_sign(float x)
{
	return x > 0 ? 1 : -1;
}

__device__ float util_GGX_P22(float slope_x, float slope_y, float alpha_x, float alpha_y)
{
	const float tmp = 1.0f + slope_x * slope_x / (alpha_x * alpha_x) + slope_y * slope_y / (alpha_y * alpha_y);
	const float value = 1.0f / (PI * alpha_x * alpha_x) / (tmp * tmp);
	return value;
}

__device__ float util_D(const glm::vec3& wm, float alpha_x, float alpha_y){
	if (wm.z <= 0.0f)
		return 0.0f;

	// slope of wm
	const float slope_x = -wm.x / wm.z;
	const float slope_y = -wm.y / wm.z;

	// value
	const float value = util_GGX_P22(slope_x, slope_y, alpha_x, alpha_y) / (wm.z * wm.z * wm.z * wm.z);
	return value;
}

__device__ float util_GGX_lambda(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	if (wi.z > 0.9999f)
		return 0.0f;
	if (wi.z < -0.9999f)
		return -1.0f;

	// a
	const float theta_i = acosf(wi.z);
	const float a = 1.0f / tanf(theta_i) / util_alpha_i(wi, alpha_x, alpha_y);

	// value
	const float value = 0.5f * (-1.0f + util_sign(a) * sqrtf(1 + 1 / (a * a)));

	return value;
}

__device__ float util_GGX_extinction_coeff(const glm::vec3& w, float alpha_x, float alpha_y)
{
	return w.z * util_GGX_lambda(w, alpha_x, alpha_y);
}



__device__ float util_GGX_projectedArea(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	if (wi.z > 0.9999f)
		return 1.0f;
	if (wi.z < -0.9999f)
		return 0.0f;

	// a
	const float theta_i = acosf(wi.z);
	const float sin_theta_i = sinf(theta_i);

	const float alphai = util_alpha_i(wi, alpha_x, alpha_y);

	// value
	const float value = 0.5f * (wi.z + sqrtf(wi.z * wi.z + sin_theta_i * sin_theta_i * alphai * alphai));

	return value;
}

__device__ glm::vec2 util_GGX_sampleP22_11(const float theta_i, const float U, const float U_2)
{
	glm::vec2 slope;

	if (theta_i < 0.0001f)
	{
		const float r = sqrtf(U / (1.0f - U));
		const float phi = 6.28318530718f * U_2;
		slope.x = r * cosf(phi);
		slope.y = r * sinf(phi);
		return slope;
	}

	// constant
	const float sin_theta_i = sinf(theta_i);
	const float cos_theta_i = cosf(theta_i);
	const float tan_theta_i = sin_theta_i / cos_theta_i;

	// slope associated to theta_i
	const float slope_i = cos_theta_i / sin_theta_i;

	// projected area
	const float projectedarea = 0.5f * (cos_theta_i + 1.0f);
	if (projectedarea < 0.0001f || projectedarea != projectedarea)
		return glm::vec2(0, 0);
	// normalization coefficient
	const float c = 1.0f / projectedarea;

	const float A = 2.0f * U / cos_theta_i / c - 1.0f;
	const float B = tan_theta_i;
	const float tmp = 1.0f / (A * A - 1.0f);

	const float D = sqrtf(glm::max(0.0f, B * B * tmp * tmp - (A * A - B * B) * tmp));
	const float slope_x_1 = B * tmp - D;
	const float slope_x_2 = B * tmp + D;
	slope.x = (A < 0.0f || slope_x_2 > 1.0f / tan_theta_i) ? slope_x_1 : slope_x_2;

	float U2;
	float S;
	if (U_2 > 0.5f)
	{
		S = 1.0f;
		U2 = 2.0f * (U_2 - 0.5f);
	}
	else
	{
		S = -1.0f;
		U2 = 2.0f * (0.5f - U_2);
	}
	const float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) / (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	slope.y = S * z * sqrtf(1.0f + slope.x * slope.x);

	return slope;
}


__device__ float util_Dwi(const glm::vec3& wi, const glm::vec3& wm, float alpha_x, float alpha_y){
	if (wm.z <= 0.0f)
		return 0.0f;

	// normalization coefficient
	const float projectedarea = util_GGX_projectedArea(wi, alpha_x, alpha_y);
	if (projectedarea == 0)
		return 0;
	const float c = 1.0f / projectedarea;

	// value
	const float value = c * glm::max(0.0f, dot(wi, wm)) * util_D(wm, alpha_x, alpha_y);
	return value;
}

__device__ glm::vec3 util_sample_D_wi(const glm::vec3& wi, const glm::vec2& rand, float alpha_x, float alpha_y)
{
	float U1 = rand.x, U2 = rand.y;
	// stretch to match configuration with alpha=1.0	
	glm::vec3 wi_11 = glm::normalize(glm::vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// sample visible slope with alpha=1.0
	glm::vec2 slope_11 = util_GGX_sampleP22_11(acosf(wi_11.z), U1, U2);

	// align with view direction
	const float phi = atan2(wi_11.y, wi_11.x);
	glm::vec2 slope(cosf(phi) * slope_11.x - sinf(phi) * slope_11.y, sinf(phi) * slope_11.x + cos(phi) * slope_11.y);

	// stretch back
	slope.x *= alpha_x;
	slope.y *= alpha_y;

	// if numerical instability
	/*if ((slope.x != slope.x) || slope.x<1e37f)
	{
		if (wi.z > 0) return glm::vec3(0.0f, 0.0f, 1.0f);
		else return glm::normalize(glm::vec3(wi.x, wi.y, 0.0f));
	}*/

	// compute normal
	const glm::vec3 wm = glm::normalize(glm::vec3(-slope.x, -slope.y, 1.0f));
	return wm;
}

__device__ inline glm::vec3 util_sample_ggx_vndf(const glm::vec3& wo, const glm::vec2& rand, float alpha_x, float alpha_y)
{
	glm::vec3 v = glm::normalize(glm::vec3(wo.x * alpha_x, wo.y * alpha_y, wo.z));
	glm::vec3 t1 = v.z > 1 - 1e-9 ? glm::vec3(1, 0, 0) : glm::cross(v, glm::vec3(0, 0, 1));
	glm::vec3 t2 = glm::cross(t1, v);
	float a = 1 / (1 + v.z);
	float r = sqrt(rand.x);
	float phi = rand.y < a ? rand.y / a * PI : ((rand.y - a) / (1.0 - a) + 1) * PI;
	float p1 = r * cos(phi);
	float p2 = r * sin(phi);
	p2 *= rand.y < a ? 1.0 : v.z;
	glm::vec3 h = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;
	return glm::normalize(glm::vec3(h.x * alpha_x, h.y * alpha_y, max(0.0f, h.z)));
}

__device__ inline float util_pow_5(float x)
{
	float x2 = x * x;
	return x2 * x2 * x;
}

//TODO: use wavelength dependent frensel 
__device__ inline glm::vec3 util_fschlick(glm::vec3 f0, glm::vec3 wi, glm::vec3 wh)
{
	float HoV = glm::max(glm::dot(wi, wh), 0.0f);
	return f0 + (1.0f - f0) * util_pow_5(1.0f - HoV);
}

__device__ glm::vec3 util_conductor_evalPhaseFunction(const glm::vec3& wi, const glm::vec3& wo, float alpha_x, float alpha_y, const glm::vec3& albedo)
{
	// half vector 
	const glm::vec3 wh = normalize(wi + wo);
	if (wh.z < 0.0f)
		return glm::vec3(0.0f);

	// value
	return 0.25f * util_Dwi(wi, wh, alpha_x, alpha_y) * util_fschlick(albedo, wi, wh) / dot(wi, wh);
}

__device__ glm::vec3 util_conductor_samplePhaseFunction(const glm::vec3& wi, const glm::vec2& random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo)
{
	glm::vec3 wh = util_sample_ggx_vndf(wi, random, alpha_x, alpha_y);

	// reflect
	glm::vec3 wo = glm::normalize(-wi + 2.0f * wh * dot(wi, wh));
	throughput *= util_fschlick(albedo, wi, wh);

	return wo;
}


__device__ glm::vec3 bxdf_asymMicrofacet_sample(glm::vec3 wo, glm::vec3& throughput, thrust::default_random_engine& rng, const asymMicrofacetInfo& mat, int order)
{
	float z = 0;
	glm::vec3 w = glm::normalize(-wo);
	int i = 0;
	thrust::uniform_real_distribution<float> u01(0, 1);
	throughput = glm::vec3(1.0f);

	while (i <= order)
	{	
		float U = u01(rng);
		float sigmaIn = max(z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA) : util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB),0.0f);
		float sigmaOut = max(z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB) : util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA),0.0f);
		float deltaZ = w.z / glm::length(w) * (-log(U) / sigmaIn);
		if (z < mat.zs != z + deltaZ < mat.zs)
		{
			deltaZ = (mat.zs - z) + (deltaZ - (mat.zs - z)) * sigmaIn / sigmaOut;
		}
		z += deltaZ;
		if (z > 0) break;
		glm::vec2 rand2 = glm::vec2(u01(rng), u01(rng));
		glm::vec3 nw;
		if (z > mat.zs)
		{
			w = util_conductor_samplePhaseFunction(-w, rand2, throughput, mat.alphaXA, mat.alphaYA, mat.albedo);
		}
		else
		{
			w = util_conductor_samplePhaseFunction(-w, rand2, throughput, mat.alphaXB, mat.alphaYB, mat.albedo);
		}
		//float val = glm::dot(nw, glm::vec3(0, 0, 1));//debug
		//return nw;
		//w = nw;
		if ((z != z) || (w.z != w.z))
			return glm::vec3(0.0f, 0.0f, 1.0f);
		i++;
	}
	return w;
}

__device__ glm::vec3 bxdf_asymMicrofacet_eval(const glm::vec3& wo, const glm::vec3& wi, const glm::ivec3& rndSeed, const asymMicrofacetInfo& mat, int order)
{
	float z = 0;
	glm::vec3 w = glm::normalize(-wo);
	glm::vec3 result = glm::vec3(0);
	glm::vec3 throughput = glm::vec3(1.0f);
	int i = 0;
	thrust::default_random_engine rng = makeSeededRandomEngine(rndSeed.x, rndSeed.y, rndSeed.z);
	thrust::uniform_real_distribution<float> u01(0, 1);
	while (i <= order)
	{
		float U = u01(rng);
		float sigmaIn = z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA) : util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB);
		float sigmaOut = z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB) : util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA);
		float deltaZ = w.z / glm::length(w) * (-log(U) / sigmaIn);
		if (z < mat.zs != z + deltaZ < mat.zs)
		{
			deltaZ = (mat.zs - z) + (deltaZ - (mat.zs - z)) * sigmaIn / sigmaOut;
		}
		z += deltaZ;
		if (z > 0) break;
		glm::vec3 p = z > mat.zs ? mat.fEval(-w, wi, mat.alphaXA, mat.alphaYA, mat.albedo) : mat.fEval(-w, wi, mat.alphaXB, mat.alphaYB, mat.albedo);
		float tau_exit = glm::max(z, mat.zs) * util_GGX_lambda(wi, mat.alphaXA, mat.alphaYA) + glm::min(z - mat.zs, 0.0f) * util_GGX_lambda(wi, mat.alphaXB, mat.alphaYB);
		result += throughput * exp(tau_exit) * p;
		glm::vec2 rand2 = glm::vec2(u01(rng), u01(rng));
		if(z> mat.zs)
		{
			w = (*mat.fSample)(-w, rand2, throughput, mat.alphaXA, mat.alphaYA, mat.albedo);
		}
		else
		{
			w = (*mat.fSample)(-w, rand2, throughput, mat.alphaXB, mat.alphaYB, mat.albedo);
		}


		if ((z != z) || (w.z != w.z))
			return glm::vec3(0.0f);
		i++;
	}
	return result;
}

__device__ inline glm::vec3 bxdf_asymMicrofacet_sample_f(const glm::vec3& wo,  glm::vec3* wi, thrust::default_random_engine& rng, float* pdf, const asymMicrofacetInfo& mat, int order)
{
	glm::vec3 throughput = glm::vec3(1.0f);
	*wi = bxdf_asymMicrofacet_sample(wo, throughput, rng, mat, order);
	*pdf = 1.0f;
	return throughput;
}