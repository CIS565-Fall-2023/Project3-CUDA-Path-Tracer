#pragma once

#include "intersections.h"

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

__host__ __device__
glm::vec3 calculateImportanceSampledDirection(
    glm::vec3 direction, float exponent, thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float theta = glm::acos(glm::pow(u01(rng), 1.f / (exponent + 1)));
    float phi = u01(rng) * TWO_PI;
    glm::vec3 tangent = glm::cross(direction, glm::vec3{0, 0, 1});
    glm::vec3 bitangent = glm::cross(direction, tangent);
    return tangent * glm::cos(phi) * glm::sin(theta) +
           bitangent * glm::sin(phi) * glm::sin(theta) +
           direction * glm::cos(theta);
}

__host__ __device__
float FresnelDielectricEvaluate(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    bool entering = cosThetaI > 0.f;
    if (!entering)
    {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = glm::abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    if (sinThetaT >= 1)
        return 1;

    float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__
float SchlickFresnel(float cosThetaI, float etaI, float etaT)
{
    float R0 = (etaI - etaT) / (etaI + etaT);
    return R0 + (1.f - R0) * glm::pow(1 - cosThetaI, 5);
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
__host__ __device__ void scatterRay(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);
    // if (m.hasReflective && m.hasRefractive)
    // {
    //     // fresnel
    //     float cosThetaI = glm::dot(-pathSegment.ray.direction, normal);
    //     p = SchlickFresnel(cosThetaI, 1.f, m.indexOfRefraction);
    // }

    if (p * m.hasReflective > (1 - p) * m.hasRefractive)
    {
        // imperfect specular lighting
        // reference: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
        pathSegment.color *= m.specular.color;
        glm::vec3 specularDirection = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.direction = calculateImportanceSampledDirection(specularDirection, m.specular.exponent, rng);
        pathSegment.ray.origin = intersect + .001f * pathSegment.ray.direction;
    }
    else if (p * m.hasReflective < (1 - p) * m.hasRefractive)
    {
        bool entering = glm::dot(-pathSegment.ray.direction, normal) > 0;
        normal *= entering ? 1.f : -1.f;
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
        // float eta = m.indexOfRefraction;
        glm::vec3 refractionDirection = glm::refract(pathSegment.ray.direction, normal, eta);

        // handle internal reflection
        if (glm::length(refractionDirection) < .01f) {
            refractionDirection = glm::reflect(pathSegment.ray.direction, normal);
        } else {
            refractionDirection = glm::normalize(refractionDirection);
        }

        pathSegment.color *= m.specular.color;
        pathSegment.ray.direction = calculateImportanceSampledDirection(refractionDirection, m.specular.exponent, rng);
        pathSegment.ray.origin = intersect + .001f * pathSegment.ray.direction;
    }
    else
    {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + .001f * pathSegment.ray.direction;
    }

    pathSegment.remainingBounces--;
}
