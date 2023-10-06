#pragma once

#include "common.h"

#include "sceneStructs.h"
#include "rng.h"

inline CPU_GPU glm::vec2 SquareToDiskConcentric(const glm::vec2& xi)
{
	glm::vec2 offset = xi * 2.f - 1.f;

	if (offset.x != 0.f || offset.y != 0.f)
	{
		float theta, r;
		if (glm::abs(offset.x) > glm::abs(offset.y))
		{
			r = offset.x;
			theta = PiOver4 * (offset.y / offset.x);
		}
		else
		{
			r = offset.y;
			theta = PiOver2 - PiOver4 * (offset.x / offset.y);
		}
		return r * glm::vec2(glm::cos(theta), glm::sin(theta));
	}
	return glm::vec2(0.);
}

inline CPU_GPU glm::vec3 SquareToHemisphereCosine(const glm::vec2& xi)
{
	glm::vec3 result = glm::vec3(SquareToDiskConcentric(xi), 0.f);
	result.z = glm::sqrt(glm::max(0.f, 1.f - result.x * result.x - result.y * result.y));
	result.z = glm::max(result.z, 0.01f);

	return result;
}

inline CPU_GPU glm::vec3 SquareToSphereUniform(const glm::vec2& xi)
{
	float z = 1.f - 2.f * xi.x;

	return glm::vec3(glm::cos(2 * Pi * xi.y) * glm::sqrt(1.f - z * z),
		glm::sin(2 * Pi * xi.y) * glm::sqrt(1.f - z * z),
		z);
}

inline CPU_GPU float SquareToSphereUniformPDF(const glm::vec3& sample)
{
	return Inv4Pi;
}

inline CPU_GPU float FresnelDielectric(const float& etaI, 
									   const float& etaO, 
									   const float& cosThetaI,
									   const float& cosThetaO)
{
	float Rparl = ((etaO * cosThetaI) - (etaI * cosThetaO)) / ((etaO * cosThetaI) + (etaI * cosThetaO));
	float Rperp = ((etaI * cosThetaI) - (etaO * cosThetaO)) / ((etaI * cosThetaI) + (etaO * cosThetaO));

	return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

inline CPU_GPU void Refract(const glm::vec3& dir, 
						  const glm::vec3& normal, 
						  const float& eta, 
						  const float& cosThetaI,
						  const float& cosThetaO,
						  glm::vec3& wiW)
{
	wiW = glm::normalize(eta * dir + (eta * cosThetaI - cosThetaO) * normal);
}

inline CPU_GPU glm::vec3 SampleGGX(CudaRNG& rng, const float& alpha)
{
	float rand_num = rng.rand();
	
	float cosTheta = std::sqrt((1.0f - rand_num) / (1.0f + (alpha * alpha - 1.0f) * rand_num));
	float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));
	
	float phi = 2.0f * Pi * rng.rand();
	glm::vec3 wh = glm::vec3(sinTheta * glm::cos(phi), 
							sinTheta * glm::sin(phi), 
							cosTheta);
	return wh;
}

inline CPU_GPU float SchlickWeight(float cos_theta)
{
	float w = glm::clamp(1.f - cos_theta, 0.f, 1.f);
	float a = w * w;
	return w * a * a;
}

inline CPU_GPU glm::vec3 FresnelSchlick(const glm::vec3& R, const float& cosTheta)
{
	return glm::mix(R, glm::vec3(1.f), SchlickWeight(cosTheta));
}

inline CPU_GPU float Lambda(const float& cos_theta, float alpha)
{
	if (cos_theta < Epsilon) return 0.f;
	float tan2_theta = cos_theta * cos_theta;
	float alpha_tan2_theta = tan2_theta * tan2_theta * alpha;
	return (-1 + sqrt(1.f + alpha_tan2_theta)) / 2;
}

inline CPU_GPU float GGX_D(const float& cos_theta, float alpha)
{
	if (cos_theta < Epsilon) return 0.f;
	float denom = (cos_theta * cos_theta) * (alpha - 1.0f) + 1.0f;
	return alpha / (Pi * denom * denom);
}

inline CPU_GPU float GGX_Pdf(const float& cos_theta, const float& cos_theta_oh, float alpha)
{
	if (cos_theta < Epsilon) return 0.f;
	return GGX_D(cos_theta, alpha) * cos_theta / (4.0f * cos_theta_oh);
}

inline CPU_GPU float GGX_G(const float& cos_theta_o, const float& cos_theta_i, float alpha)
{
	return 1.f / (1.f + Lambda(cos_theta_o, alpha) + Lambda(cos_theta_i, alpha));
}

class EvaluateBSDF
{
public:
	inline static GPU_ONLY glm::vec3 MicrofacetReflection(const glm::vec3& albedo, 
														  const float& cos_theta_i, 
														  const float& cos_theta_o,
														  const float& cos_theta_h,
														  const float& cos_theta_oh,
														  const float& metallic,
														  const float& alpha)
	{
		if (cos_theta_i * cos_theta_o < Epsilon) return glm::vec3(0.f);
		float D = GGX_D(cos_theta_h, alpha);
		float G = GGX_G(cos_theta_o, cos_theta_i, alpha);
		return albedo * D * G / (4.f * cos_theta_i * cos_theta_o);
	}

	inline static GPU_ONLY glm::vec3 MicrofacetMix(const glm::vec3& albedo,
		const float& cos_theta_i,
		const float& cos_theta_o,
		const float& cos_theta_h,
		const float& cos_theta_oh,
		const float& metallic,
		const float& alpha)
	{
		if (cos_theta_i * cos_theta_o < Epsilon) return glm::vec3(0.f);

		glm::vec3 F = FresnelSchlick(glm::mix(glm::vec3(0.04f), albedo, metallic), cos_theta_oh);
		float D = GGX_D(cos_theta_h, alpha);
		float G = GGX_G(cos_theta_o, cos_theta_i, alpha);
		return glm::mix(albedo * InvPi * (1.f - metallic), glm::vec3(D * G / (4.f * cos_theta_i * cos_theta_o)), F);
	}
};

class SampleBSDF
{
public:
	inline static GPU_ONLY bool Sample(const Material& material,
									const ShadeableIntersection& intersection,
									CudaRNG& rng,
									BSDFSample& sample,
									const float& metallic, 
									const float& roughness)
	{
		// woW is stored in sample.wiW
		switch (material.type & MaterialType::Clear_Texture)
		{
		case MaterialType::DiffuseReflection:
		{
			return DiffuseReflection(material.GetAlbedo(intersection.uv), intersection, rng, sample);
		}
		case MaterialType::SpecularReflection:
		{
			return SpecularReflection(material.GetAlbedo(intersection.uv), intersection, sample);
		}
		case MaterialType::SpecularGlass:
		{
			return SpecularGlass(material.GetAlbedo(intersection.uv), ETA_AIR, material.eta, intersection, rng, sample);
		}
		case MaterialType::SubsurfaceScattering:
		{
			return SubsurfaceScattering(material.GetAlbedo(intersection.uv), 0, intersection, rng, sample);
		}
		case MaterialType::MicrofacetReflection:
		{
			return MicrofacetReflection(material.GetAlbedo(intersection.uv), intersection, rng, sample, metallic, roughness);
		}
		case MaterialType::MicrofacetMix:
		{
			return MicrofacetMix(material.GetAlbedo(intersection.uv), intersection, rng, sample, metallic, roughness);
		}
		}
		return false;
	}

protected:
	inline static GPU_ONLY bool DiffuseReflection(const glm::vec3& albedo,
												const ShadeableIntersection& intersection,
												CudaRNG& rng,
												BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		if (glm::dot(intersection.normal, sample.wiW) < 0.f)
		{
			return false;
		}
		sample.f	= albedo;
		sample.wiW	= SquareToHemisphereCosine({ rng.rand(), rng.rand() });
		sample.wiW	= glm::normalize(LocalToWorld(intersection.normal) * sample.wiW);
		sample.pdf	= glm::dot(sample.wiW, intersection.normal);
		return true;
	}

	inline static GPU_ONLY bool SpecularReflection(const glm::vec3& albedo,
												const ShadeableIntersection& intersection,
												BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		if (glm::dot(intersection.normal, sample.wiW) < 0.f)
		{
			return false;
		}
		sample.wiW = glm::normalize(glm::reflect(-sample.wiW, intersection.normal));
		sample.f = albedo / glm::abs(glm::dot(sample.wiW, intersection.normal));
		sample.pdf = 1.f;
		return true;
	}

	inline static GPU_ONLY bool SpecularGlass(const glm::vec3& albedo,
									float etaA,
									float etaB,
									const ShadeableIntersection& intersection,
									CudaRNG& rng,
									BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		float cosThetaI = glm::dot(intersection.normal, sample.wiW);
		glm::vec3 normal = intersection.normal;
		if (cosThetaI < 0.f)
		{
			normal = -normal;
			cosThetaI = -cosThetaI;
			thrust::swap(etaA, etaB);
		}
		float eta = etaA / etaB;

		float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
		float sinThetaO = eta * sinThetaI;
		float cosThetaO = glm::sqrt(glm::max(0.f, 1.f - sinThetaO * sinThetaO));

		float F = 1.f;

		if (sinThetaO < 1.f)
		{
			F = FresnelDielectric(etaA, etaB, cosThetaI, cosThetaO);
		}

		if (rng.rand() < F) // Reflection
		{
			SpecularReflection(albedo, intersection, sample);
		}
		else // Refraction
		{
			Refract(-sample.wiW, normal, eta, cosThetaI, cosThetaO, sample.wiW);
			sample.f = albedo / glm::abs(glm::dot(sample.wiW, intersection.normal));
			sample.pdf = 1.f;
		}
		return true;
	}

	inline static GPU_ONLY bool MicrofacetReflection(const glm::vec3& albedo,
													 const ShadeableIntersection& intersection,
													 CudaRNG& rng,
													 BSDFSample& sample,
													 const float& metallic,
													 const float& roughness)
	{
		// woW is stored in sample.wiW
		float cos_theta_o = glm::dot(sample.wiW, intersection.normal);
		if (cos_theta_o < 0.f)
		{
			return false;
		}

		float alpha = roughness * roughness;
		glm::vec3 wh = SampleGGX(rng, alpha);
		if (glm::abs(wh.z) < Epsilon) return false;
		
		glm::vec3 whW = glm::normalize(LocalToWorld(intersection.normal) * wh);
		glm::vec3 wiW = glm::normalize(glm::reflect(-sample.wiW, whW));
		float cos_theta_i = glm::dot(wiW, intersection.normal);
		if (cos_theta_i < 0.f)
		{
			return false;
		}
		
		float cos_theta_oh	= glm::dot(sample.wiW, whW);
		
		alpha = glm::max(alpha, 0.09f);
		
		sample.f = EvaluateBSDF::MicrofacetReflection(albedo,
														cos_theta_i,
														cos_theta_o,
														wh.z,
														cos_theta_oh,
														1.f, alpha);

		sample.pdf = GGX_Pdf(wh.z, cos_theta_oh, alpha);
		if (sample.pdf < 0.1f)
		{
			sample.f = albedo;
			sample.pdf = glm::dot(wiW, intersection.normal);
		}
		sample.wiW = wiW;
		return true;
	}

	inline static GPU_ONLY bool MicrofacetMix(const glm::vec3& albedo,
												const ShadeableIntersection& intersection,
												CudaRNG& rng,
												BSDFSample& sample,
												const float& metallic,
												const float& roughness)
	{
		// woW is stored in sample.wiW
		float cos_theta_o = glm::dot(sample.wiW, intersection.normal);
		if (cos_theta_o < 0.f)
		{
			return false;
		}
		float alpha = roughness * roughness;
		
		glm::vec3 wiW, whW;
		float microfacet_pdf;
		float u = 1.f / (2.f - metallic);
		if (rng.rand() > u)
		{
			glm::vec3 wi = SquareToHemisphereCosine({ rng.rand(), rng.rand() });
			wiW = glm::normalize(LocalToWorld(intersection.normal) * wi);
			whW = glm::normalize(sample.wiW + wiW);
		}
		else
		{
			glm::vec3 wh = SampleGGX(rng, alpha);
			if (glm::abs(wh.z) < Epsilon) return false;
		
			whW = glm::normalize(LocalToWorld(intersection.normal) * wh);
			wiW = glm::normalize(glm::reflect(-sample.wiW, whW));
		
			float cos_theta_oh = glm::dot(sample.wiW, whW);
			microfacet_pdf = GGX_Pdf(wh.z, cos_theta_oh, alpha);
		}

		float cos_theta_i = glm::dot(wiW, intersection.normal);
		
		if (cos_theta_i < 0.f)
		{
			return false;
		}
		alpha = glm::max(alpha, 0.09f);
		
		float diffuse_pdf = cos_theta_i *InvPi;

		float cos_theta_oh = glm::dot(sample.wiW, whW);
		float cos_theta_h = glm::dot(whW, intersection.normal);
		
		microfacet_pdf = GGX_Pdf(cos_theta_h, cos_theta_oh, alpha);

		sample.f =  EvaluateBSDF::MicrofacetMix(albedo,
											   cos_theta_i,
											   cos_theta_o,
											   cos_theta_h,
											   cos_theta_oh,
											   metallic, alpha);

		sample.pdf = glm::mix(diffuse_pdf, microfacet_pdf, u);

		if (sample.pdf < 0.1f)
		{
			sample.f = albedo;
			sample.pdf = glm::dot(wiW, intersection.normal);
		}
		sample.wiW = wiW;
		return true;
	}

	inline static GPU_ONLY bool SubsurfaceScattering(const glm::vec3& albedo, 
													float alpha,
													const ShadeableIntersection& intersection,
													CudaRNG& rng,
													BSDFSample& sample)
	{
		return false;
	}
};
