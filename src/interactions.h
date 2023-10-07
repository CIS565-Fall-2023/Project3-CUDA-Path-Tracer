#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrtf(u01(rng)); // cos(theta)
    float over = sqrtf(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (fabsf(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (fabsf(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cosf(around) * over * perpendicularDirection1
        + sinf(around) * over * perpendicularDirection2;
}

// reference: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
__device__
glm::vec3 calculateRandomDirectionWithImportanceSampling(
    glm::vec3 direction, float exponent, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float theta = acosf(powf(u01(rng), 1.f / (exponent + 1)));
    float phi = u01(rng) * TWO_PI;

    glm::vec3 differentDirection;
    if (fabsf(direction.x) < SQRT_OF_ONE_THIRD) {
        differentDirection = glm::vec3(1, 0, 0);
    }
    else if (fabsf(direction.y) < SQRT_OF_ONE_THIRD) {
        differentDirection = glm::vec3(0, 1, 0);
    }
    else {
        differentDirection = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(direction, differentDirection));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(direction, perpendicularDirection1));

    return direction * cosf(theta) +
        perpendicularDirection1 * sinf(theta) * cosf(phi) +
        perpendicularDirection2 * sinf(theta) * sinf(phi);
}

__device__
void DiffuseBxDF(
    PathSegment& pathSegment,
    glm::vec3& intersect,
    glm::vec3& normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
    normal = entering ? normal : -normal;
    pathSegment.ray.direction = glm::normalize(
        calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.color *= m.albedo;
}

__device__
void SpecularBxDF(
    PathSegment& pathSegment,
    glm::vec3& intersect,
    glm::vec3& normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
    normal = entering ? normal : -normal;
    glm::vec3 reflectDirection = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.direction = glm::normalize(
        calculateRandomDirectionWithImportanceSampling(
            reflectDirection, powf(m.roughness + EPSILON, -2), rng));
    // reflection light may go into opaque object, but current method seems fine
    if (glm::dot(pathSegment.ray.direction, normal) < 0)
    {
        pathSegment.color *= 1 - m.opacity;
    }
}

__device__
void TransmissionBxDF(
    PathSegment& pathSegment,
    glm::vec3& intersect,
    glm::vec3& normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
    normal = entering ? normal : -normal;
    float eta = entering ? 1.f / m.ior : m.ior;
    glm::vec3 refractionDirection = glm::refract(pathSegment.ray.direction, normal, eta);

    // total internal reflection
    if (glm::length(refractionDirection) < .01f)
    {
        SpecularBxDF(pathSegment, intersect, normal, m, rng);
    }
    else
    {
        refractionDirection = glm::normalize(refractionDirection);
        pathSegment.ray.direction = glm::normalize(
            calculateRandomDirectionWithImportanceSampling(
                refractionDirection, powf(m.roughness + EPSILON, -2), rng));
        pathSegment.color *= m.albedo;
    }
}

__device__
inline void ConductorFresnel(
    PathSegment& pathSegment,
    const glm::vec3& normal,
    const Material& m)
{
    float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
    pathSegment.color *= m.albedo + (glm::vec3(1.f) - m.albedo) * glm::clamp(powf(1 - fabsf(cosThetaI), 5), .0f, .99f);
}

__device__
inline float DielectricFresnel(
    const PathSegment& pathSegment,
    const glm::vec3& normal,
    const Material& m)
{
    float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
    bool entering = cosThetaI > 0;
    float eta = entering ? 1.f / m.ior : m.ior;
    float R0 = powf((eta - 1) / (eta + 1), 2);
    float R = R0 + (1 - R0) * glm::clamp(powf(1 - fabsf(cosThetaI), 5), .0f, .99f);
    return R;
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
__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // compensate for get point on ray
    // intersect += .0001f * pathSegment.ray.direction;
    // uniform distribution
    thrust::uniform_real_distribution<float> u01(0, 1);
    // metal
    if (u01(rng) < m.metallic)
    {
        // conductor fresnel
        ConductorFresnel(pathSegment, normal, m);
        SpecularBxDF(pathSegment, intersect, normal, m, rng);
    }
    // dielectric
    else
    {
        // fresnel
        if (u01(rng) < DielectricFresnel(pathSegment, normal, m))
        {
            // reflection
            SpecularBxDF(pathSegment, intersect, normal, m, rng);
        }
        else
        {
            if (u01(rng) < m.opacity)
            {
                // diffuse
                DiffuseBxDF(pathSegment, intersect, normal, m, rng);
            }
            else
            {
                // refraction
                TransmissionBxDF(pathSegment, intersect, normal, m, rng);
            }
        }
    }
    pathSegment.ray.origin = intersect + .002f * pathSegment.ray.direction;
    pathSegment.remainingBounces--;
}


// some procedural texture
namespace ProceduralTexture
{
__device__ glm::vec3 palettes(glm::vec2 uv)
{
    glm::vec3 a(0.5, 0.5, 0.5), b(0.5, 0.5, 0.5), c(1.0, 1.0, 1.0), d(0.00, 0.33, 0.67);
    return a + b * glm::cos(TWO_PI * (c * glm::length(uv) + d));
}
__device__ glm::vec3 checkerboard(glm::vec2 uv)
{
    if ((int)(uv.x * 10) % 2 == (int)(uv.y * 10) % 2)
        return glm::vec3(.2f);
    else
        return glm::vec3(.8f);
}
__device__ glm::vec3 random(glm::vec2 uv)
{
    thrust::default_random_engine rng(int(uv.x * 128) * 128 + int(uv.y * 128));
    float r = thrust::uniform_real_distribution<float>(0, 1)(rng);
    float f = 0.5f + 0.5f * glm::sin(uv.x * 10.f * r * TWO_PI);
    float g = 0.5f + 0.5f * glm::cos(uv.y * 10.f * r * TWO_PI);
    return glm::vec3(f * g);
}
} // namespace ProceduralTexture
