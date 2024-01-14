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

__device__ inline void util_math_get_TBN_naive(const glm::vec3& N, glm::vec3* T, glm::vec3* B)
{
	glm::vec3 tmp = abs(N.x) > 1.0 - EPSILON ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
	*T = glm::cross(N, tmp);
	*B = glm::cross(N, *T);
}

__device__ inline glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random)
{
	float z = random.x;
	float sq_1_z_2 = sqrt(max(1 - z * z, 0.0f));
	float phi = TWO_PI * random.y;
	return glm::vec3(cos(phi) * sq_1_z_2, sin(phi) * sq_1_z_2, z);
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

//https://jcgt.org/published/0007/04/01/paper.pdf
__device__ inline glm::vec3 util_math_sample_ggx_vndf(const glm::vec3& wo, float roughness, const glm::vec2& rand)
{
	glm::vec3 v = glm::normalize(glm::vec3(wo.x * roughness, wo.y * roughness, wo.z));
	glm::vec3 t1 = v.z > 1 - 1e-9 ? glm::vec3(1, 0, 0) : glm::cross(v, glm::vec3(0, 0, 1));
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
	return (a2 + 1e-10f) / (denom + 1e-10f);
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

__device__ glm::vec3 bxdf_perfect_specular_sample_f(const glm::vec3& wo, glm::vec3* wi, float* pdf, glm::vec3 reflectionAlbedo)
{
	*wi = glm::vec3(-wo.x, -wo.y, wo.z);
	*pdf = 1;
	return reflectionAlbedo / util_math_tangent_space_abscos(wo);
}

__device__ inline glm::vec3 bxdf_microfacet_eval(const glm::vec3& wo, const glm::vec3& wi, glm::vec3& baseColor, float roughness)
{
	glm::vec3 wh = glm::normalize(wo + wi);
	float a2 = roughness * roughness;
	glm::vec3 F = util_math_fschlick(baseColor, glm::abs(glm::dot(wo, wh)));
	float D = util_math_ggx_normal_distribution(wh, a2);
	float G2 = util_math_smith_ggx_shadowing_masking(wi, wo, a2);
	return (F * G2 * D) / (max(4 * util_math_tangent_space_abscos(wo) * util_math_tangent_space_abscos(wi), 1e-9f));
}

__device__ inline float bxdf_microfacet_pdf(const glm::vec3& wo, const glm::vec3& wi, float roughness)
{
	float a2 = roughness * roughness;
	glm::vec3 wh = glm::normalize(wo + wi);
	float G1 = util_math_smith_ggx_masking(wo, a2);
	float D = util_math_ggx_normal_distribution(wh, a2);

	return (G1 * D) / (max(4 * util_math_tangent_space_abscos(wo), 1e-9f));
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
	return bxdf_microfacet_eval(wo, *wi, reflectionAlbedo, roughness);
}

__device__ inline float util_math_calc_lS(float metallic)
{
	return 1.0 / (6.0 - 5.0 * sqrt(metallic));
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
	if (wi->z < 0.0)
	{
		*pdf = 0;
		return glm::vec3(0.0);
	}
	else
	{
		*pdf = bxdf_metallic_workflow_pdf(wo, *wi, baseColor, metallic, roughness);
		return bxdf_metallic_workflow_eval(wo, *wi, baseColor, metallic, roughness);
	}
} 

// https://www.shadertoy.com/view/lsV3zV
__device__ inline glm::vec3 bxdf_blinn_phong_eval(const glm::vec3& wo, const glm::vec3& wi, glm::vec3& baseColor, float specExponent)
{
	float cosThetaN_Wi = util_math_tangent_space_abscos(wi);
	float cosThetaN_Wo = util_math_tangent_space_abscos(wo);
	glm::vec3 wh = normalize(wi + wo);
	float cosThetaN_Wh = util_math_tangent_space_abscos(wh);

	// Compute geometric term of blinn microfacet      
	float cosThetaWo_Wh = abs(dot(wo, wh));
	float G = min(1., min((2. * cosThetaN_Wh * cosThetaN_Wo / cosThetaWo_Wh),
		(2. * cosThetaN_Wh * cosThetaN_Wi / cosThetaWo_Wh)));

	// Compute distribution term
	float D = (specExponent + 2.) * INV_TWO_PI * pow(max(0.0f, cosThetaN_Wh), specExponent);

	// assume no fresnel
	float F = 1.;

	return baseColor * D * G * F / (4.0f * cosThetaN_Wi * cosThetaN_Wo);
}

__device__ inline float bxdf_blinn_phong_pdf(const glm::vec3& wo, const glm::vec3& wi, float specExponent)
{
	glm::vec3 wh = normalize(wi + wo);
	float cosTheta = util_math_tangent_space_abscos(wh);

	float pdf = 0.0f;
	if (dot(wo, wh) > 0.)
	{
		pdf = ((specExponent + 1.0f) * pow(max(0.0f, cosTheta), specExponent)) / (TWO_PI * 4. * dot(wo, wh));
	}

	return pdf;
}

__device__ inline glm::vec3 bxdf_blinn_phong_sample(const glm::vec2& random, const glm::vec3& wo, float specExponent)
{
	float cosTheta = pow(max(0.0f, random.x), 1.0f / (specExponent + 1.0f));
	float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	float phi = random.y * TWO_PI;

	glm::vec3 wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	glm::vec3 wi = glm::reflect(-wo, wh);
	return wi;
}

__device__ inline glm::vec3 bxdf_blinn_phong_sample_f(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& random, float* pdf, glm::vec3 baseColor, float specExponent)
{
	*wi = bxdf_blinn_phong_sample(random, wo, specExponent);
	*pdf = bxdf_blinn_phong_pdf(wo, *wi, specExponent);
	return bxdf_blinn_phong_eval(wo, *wi, baseColor, specExponent);
}

__device__ glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand)
{
	float t = sqrt(rand.x);
	return glm::vec2(1 - t, t * rand.y);
}

__device__ inline float util_math_solid_angle_to_area(glm::vec3 surfacePos, glm::vec3 surfaceNormal, glm::vec3 receivePos)
{
	glm::vec3 L = receivePos - surfacePos;
	return glm::abs(glm::dot(glm::normalize(L), surfaceNormal)) / glm::distance2(surfacePos, receivePos);
}

__device__ inline float util_mis_weight_balanced(float pdf1, float pdf2)
{
	return pdf1 / (pdf1 + pdf2);
}

__device__ float lights_sample_pdf(const SceneInfoDev& sceneInfo, int lightPrimID)
{
	if(!sceneInfo.lightsSize) return -1;
	float prob = 1.0 / sceneInfo.lightsSize;
	Primitive& lightPrim = sceneInfo.dev_primitives[lightPrimID];
	Object& lightObj = sceneInfo.dev_objs[lightPrim.objID];
	if (lightObj.type == GeomType::SPHERE)
	{
		glm::vec3 rWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.5, 0.0, 0.0), 0.0f));
		float R = glm::length(rWorld);
		return prob / (TWO_PI * R * R);
	}
	else if(lightObj.type == GeomType::CUBE)
	{
		glm::vec3 vx = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(1, 0, 0), 0.0f));
		glm::vec3 vy = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 1, 0), 0.0f));
		glm::vec3 vz = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 0, 1), 0.0f));
		float Axy = abs(glm::length(glm::cross(vx, vy)));
		float Axz = abs(glm::length(glm::cross(vx, vz)));
		float Ayz = abs(glm::length(glm::cross(vy, vz)));
		float area = 2 * (Axy + Axz + Ayz);
		return prob / area;
	}
	else
	{
		glm::ivec3 tri = sceneInfo.modelInfo.dev_triangles[lightObj.triangleStart + lightPrim.offset];
		const glm::vec3& v0 = sceneInfo.modelInfo.dev_vertices[tri[0]];
		const glm::vec3& v1 = sceneInfo.modelInfo.dev_vertices[tri[1]];
		const glm::vec3& v2 = sceneInfo.modelInfo.dev_vertices[tri[2]];
		glm::vec3 v0w = multiplyMV(lightObj.Transform.transform, glm::vec4(v0, 1.0f));
		glm::vec3 v1w = multiplyMV(lightObj.Transform.transform, glm::vec4(v1, 1.0f));
		glm::vec3 v2w = multiplyMV(lightObj.Transform.transform, glm::vec4(v2, 1.0f));
		glm::vec3 nNormal = glm::cross(v1w - v0w, v2w - v0w);
		float area = abs(glm::length(nNormal)) / 2;
		return prob / area;
	}
}

__device__ void lights_sample(const SceneInfoDev& sceneInfo, const glm::vec3& random, const glm::vec3& position, const glm::vec3& normal, glm::vec3* lightPos, glm::vec3* lightNormal, glm::vec3* emissive, float* pdf)
{
	if (!sceneInfo.lightsSize) return;
	float fchosen = random.x * sceneInfo.lightsSize;
	int chosenLight = std::floor(fchosen);
	float u = glm::clamp(fchosen - (int)fchosen, 0.0f, 1.0f);
	Primitive& lightPrim = sceneInfo.dev_lights[chosenLight];
	Object& lightObj = sceneInfo.dev_objs[lightPrim.objID];
	Material& lightMat = sceneInfo.dev_materials[lightObj.materialid];
	float prob = 1.0f / sceneInfo.lightsSize;
	if (lightObj.type == GeomType::SPHERE)//Assume uniform scale of xyz
	{
		glm::vec3 originWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.0), 1.0f));
		glm::vec3 rWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.5, 0.0, 0.0), 0.0f));
		float R = glm::length(rWorld);
		glm::vec3 localSample = util_sample_hemisphere_uniform(glm::vec2(random.y, random.z));
		glm::vec3 N = glm::normalize(position - originWorld);
		glm::vec3 T, B;
		util_math_get_TBN_pixar(N, &T, &B);
		*lightNormal = glm::normalize(glm::mat3(T, B, N) * localSample);
		*lightPos = originWorld + *lightNormal * R;
		prob /= (TWO_PI * R * R);
	}
	else if(lightObj.type == GeomType::CUBE)//TODO: use quad light to replace all cubes
	{
		glm::vec3 v0 = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(-0.5f, -0.5f, -0.5f), 1.0f));
		glm::vec3 vx = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(1,0,0), 0.0f));
		glm::vec3 vy = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0,1,0), 0.0f));
		glm::vec3 vz = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0,0,1), 0.0f));

		float Axy = abs(glm::length(glm::cross(vx, vy)));
		float Axz = abs(glm::length(glm::cross(vx, vz)));
		float Ayz = abs(glm::length(glm::cross(vy, vz)));
		float area = 2 * (Axy + Axz + Ayz);
		prob /= area;
		float s = u * area;
		double limit = Axy;
		float i = random.y, j = random.z;
		if (s < limit) {
			*lightPos = v0 + vx * i + vy * j;
			*lightNormal = glm::normalize(-vz);
		}
		else if (s < (limit += Axy)) {
			*lightPos = v0 + vz + vx * i + vy * j;
			*lightNormal = glm::normalize(vz);
		}
		else if (s < (limit += Axz)) {
			*lightPos = v0 + vx * i + vz * j;
			*lightNormal = glm::normalize(-vy);
		}
		else if (s < (limit += Axz)) {
			*lightPos = v0 + vy + vx * i + vz * j;
			*lightNormal = glm::normalize(vy);
		}
		else if (s < (limit += Ayz)) {
			*lightPos = v0 + vy * i + vz * j;
			*lightNormal = glm::normalize(-vx);
		}
		else {
			*lightPos = v0 + vx + vy * i + vz * j;
			*lightNormal = glm::normalize(vx);
		}
	}
	else //Triangle
	{
		glm::ivec3 tri = sceneInfo.modelInfo.dev_triangles[lightObj.triangleStart + lightPrim.offset];
		glm::vec2 bary = util_math_uniform_sample_triangle(glm::vec2(random.y, random.z));
		const glm::vec3& v0 = sceneInfo.modelInfo.dev_vertices[tri[0]];
		const glm::vec3& v1 = sceneInfo.modelInfo.dev_vertices[tri[1]];
		const glm::vec3& v2 = sceneInfo.modelInfo.dev_vertices[tri[2]];
		glm::vec3 v0w = multiplyMV(lightObj.Transform.transform, glm::vec4(v0, 1.0f));
		glm::vec3 v1w = multiplyMV(lightObj.Transform.transform, glm::vec4(v1, 1.0f));
		glm::vec3 v2w = multiplyMV(lightObj.Transform.transform, glm::vec4(v2, 1.0f));
		*lightPos = v0w * bary[0] + v1w * bary[1] + v2w * (1 - bary[0] - bary[1]);
		glm::vec3 nNormal = glm::cross(v1w - v0w, v2w - v0w);
		float area = abs(glm::length(nNormal)) / 2;
		*lightNormal = nNormal / (area > 0.0 ? area : 1e-8f);
		prob /= area;
	}
	bool vis = glm::dot(position - *lightPos, *lightNormal) > 0;
	vis = vis && glm::dot(*lightPos - position, normal) > 0;
	glm::vec3 shadowRayOri = position;
	//shadowRayOri += glm::dot(*lightPos - position, normal) > 0.0 ? normal * 0.00001f : -normal * 0.00001f;
#if USE_BVH
	vis = vis && util_bvh_test_visibility(shadowRayOri, *lightPos, sceneInfo);
#else
	vis = vis && util_test_visibility(shadowRayOri, *lightPos, sceneInfo);
#endif
	
	*pdf = prob;
	*emissive = vis ? lightMat.emittance * lightMat.color : glm::vec3(0);
}


