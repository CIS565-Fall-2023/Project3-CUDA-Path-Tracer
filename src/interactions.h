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

inline CPU_GPU float SquareToHemisphereCosinePDF(const glm::vec3& sample)
{
	return sample.z * InvPi; // cos(theta) / PI
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

inline GPU_ONLY void SampleLight()
{

}

class SampleBSDF
{
public:
	inline static GPU_ONLY void Sample(const Material& material,
									const ShadeableIntersection& intersection,
									const float& etaA,
									CudaRNG& rng,
									BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		switch (material.type & MaterialType::Clear_Texture)
		{
		case MaterialType::DiffuseReflection:
		{
			DiffuseReflection(material.GetAlbedo(intersection.uv), intersection, rng, sample);
			break;
		}
		case MaterialType::SpecularReflection:
		{
			SpecularReflection(material.GetAlbedo(intersection.uv), intersection, sample);
			break;
		}
		case MaterialType::Glass:
		{
			Glass(material.GetAlbedo(intersection.uv), ETA_AIR, material.eta, intersection, rng, sample);
			break;
		}
		}
	}

protected:
	inline static GPU_ONLY void DiffuseReflection(const glm::vec3& albedo,
												const ShadeableIntersection& intersection,
												CudaRNG& rng,
												BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		if (glm::dot(intersection.normal, sample.wiW) < 0.f)
		{
			return;
		}
		sample.f	= albedo * InvPi;
		sample.wiW	= SquareToHemisphereCosine({ rng.rand(), rng.rand() });
		sample.pdf	= SquareToHemisphereCosinePDF(sample.wiW);
		sample.wiW	= glm::normalize(LocalToWorld(intersection.normal) * sample.wiW);
	}

	inline static GPU_ONLY void SpecularReflection(const glm::vec3& albedo,
												const ShadeableIntersection& intersection,
												BSDFSample& sample)
	{
		// woW is stored in sample.wiW
		if (glm::dot(intersection.normal, sample.wiW) < 0.f)
		{
			return;
		}
		sample.wiW = glm::normalize(glm::reflect(-sample.wiW, intersection.normal));
		sample.f = albedo / glm::abs(glm::dot(sample.wiW, intersection.normal));
		sample.pdf = 1.f;
	}

	inline static GPU_ONLY void Glass(const glm::vec3& albedo,
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
	}
};
