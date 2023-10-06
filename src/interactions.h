#pragma once

#include "intersections.h"
#include "perlin.h"

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 random_in_unit_sphere(thrust::default_random_engine& rng) {
	glm::vec3 p;
	thrust::uniform_real_distribution<float> u01(0, 1);
	do {
		p = 2.0f * glm::vec3(u01(rng), u01(rng), u01(rng)) - glm::vec3(1, 1, 1);
	} while (glm::length2(p) >= 1.0f);
	return p;
}

// This is the reflection function from the pseudocode
__host__ __device__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
	return v - 2.0f * glm::dot(v, n) * n;
}

__host__ __device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * std::powf((1.0f - cosine), 5.0f);
}

__host__ __device__ bool refract(const glm::vec3& v, const glm::vec3& n, float niOverNt, glm::vec3& refracted) {
	glm::vec3 uv = glm::normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);

	if (discriminant > 0.0f) 
    {
		refracted = niOverNt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	return false;
}

__host__ __device__ glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) 
{
	auto cos_theta = std::fminf(glm::dot(-uv, n), 1.0f);

    glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = -std::sqrtf(fabs(1.0 - glm::dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__device__
void scatterRay(
        PathSegment& pathSegment,
		ShadeableIntersection intersection,
        glm::vec3 intersect,
        glm::vec3 normal,
        bool frontFace,
        const Material &material,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

	pathSegment.needSkyboxColor = false;

    switch (material.type)
    {
    case MaterialType::Emissive:
		pathSegment.ray.origin = intersect + normal * 0.001f;
		pathSegment.color *= (material.color * material.emittance);
		pathSegment.remainingBounces = 0;
        break;
    case MaterialType::Diffuse:
	{
		pathSegment.ray.origin = intersect + normal * 0.001f;

		glm::vec3 target = intersect + normal + random_in_unit_sphere(rng);
		pathSegment.ray.direction = target - intersect;
		pathSegment.color *= material.color;
		pathSegment.remainingBounces--;
	}
    break;
    case MaterialType::Metal:
	{
		pathSegment.ray.origin = intersect + normal * 0.001f;

		glm::vec3 reflected = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
		pathSegment.ray.direction = reflected + material.fuzz * random_in_unit_sphere(rng);
		pathSegment.color *= material.color * material.hasReflective;
		pathSegment.needSkyboxColor = true;
		pathSegment.remainingBounces--;
	}
	break;
    case MaterialType::Glass:
    {
		pathSegment.ray.origin = intersect;

		float refractionIndex = frontFace ? (1.0f / material.indexOfRefraction) : material.indexOfRefraction;

		//pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);

		glm::vec3 unitDirection = glm::normalize(pathSegment.ray.direction);
		float costTheta = std::fminf(glm::dot(-unitDirection, normal), 1.0f);
		float sinTheta = std::sqrtf(1.0f - costTheta * costTheta);

		bool cannotRefract = refractionIndex * sinTheta > 1.0f;

		glm::vec3 direction;

		if (cannotRefract || schlick(costTheta, refractionIndex) > u01(rng))
		{
			direction = glm::reflect(unitDirection, normal);
		}
		else
		{
			direction = glm::refract(unitDirection, normal, refractionIndex);
		}

		pathSegment.ray.direction = direction;
		pathSegment.needSkyboxColor = true;
		pathSegment.remainingBounces--;
    }
    break;
	case MaterialType::Image:
	{
		float4 color = tex2D<float4>(material.albedo, intersection.u, intersection.v);
		pathSegment.color = { color.x, color.y, color.z };
		pathSegment.remainingBounces = 0;
	}
	break;
    default:
        break;
    }

	if (material.pattern == Pattern::Ring)
	{
		float sqrt = glm::sqrt(intersect.x * intersect.x + intersect.y * intersect.y) * 4.0f;
		float floor = glm::floor(sqrt);
		if (glm::mod(glm::floor(sqrt), 2.0f))
		{
			pathSegment.color *= glm::vec3(0.0f);
		}
		else
		{
			pathSegment.color *= glm::vec3(1.0f);
		}
	}
	else if (material.pattern == Pattern::CheckerBoard)
	{
		if (glm::mod(glm::floor(intersect.x) + glm::floor(intersect.y) + glm::floor(intersect.z), 2.0f) == 0)
		{
			pathSegment.color *= glm::vec3(0.0f);
		}
		else
		{
			pathSegment.color *= glm::vec3(1.0f);
		}
	}
}
