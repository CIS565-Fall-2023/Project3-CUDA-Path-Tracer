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

__host__ __device__ void sample_f_diffuse(glm::vec3 &wi,
                                            PathSegment &pathSegment,
                                            const glm::vec3 &normal, 
                                            const Material &mat,
                                            float diffusePDF,
                                            thrust::default_random_engine& rng)
{
    wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.accum_throughput *= mat.color / diffusePDF;        // throughput
}

__host__ __device__ void sample_f_specular(glm::vec3& wi,
                                            PathSegment& pathSegment,
                                            const glm::vec3& normal,
                                            const Material& mat,
                                            float specularPDF,
                                            thrust::default_random_engine& rng)
{
    wi = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.accum_throughput *= mat.specular.color / specularPDF;      // technically PDF for specular reflections is a delta distribution
                                                                // and the "usable" ray that is a perfect reflection
                                                                // will always have a PDF of 1 since that's the only ray we can use
                                                                // but we're using this weird method of using intensities, so we'll go with it
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

__host__ __device__ void sample_f(PathSegment &pathSegment,
                                  const glm::vec3 &intersectionPoint,
                                  const glm::vec3 &normal,
                                  const Material &mat,
                                  thrust::default_random_engine &rng) {
    // Renaming this to sample_f() because we're sampling a new "bounce" (new wi) and also computing bsdf at the intersection point

    // Using squared length to avoid sqrt calculations
    // More performant without any accuracy loss
    float diffuseAmt = glm::length2(mat.color);
    float specAmt = glm::length2(mat.specular.color);

    float totalAmt = diffuseAmt + specAmt;
    float diffuseProbability = diffuseAmt / totalAmt;
    float specProbability = specAmt / totalAmt;

    glm::vec3 wi(0.0f); // next iteration ray

    thrust::uniform_real_distribution<float> u01(0, 1);

    float probability = u01(rng);
    if (probability <= diffuseProbability)
    {
        sample_f_diffuse(wi, pathSegment, normal, mat, diffuseProbability, rng);
    }
    else
    {
        sample_f_specular(wi, pathSegment, normal, mat, specProbability, rng);
    }

    pathSegment.ray.origin = intersectionPoint;
    pathSegment.ray.direction = wi;
    pathSegment.remainingBounces--;
}
