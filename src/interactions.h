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
glm::vec3 sampleDiffuse(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3& wi) { // write out to wi

    wi = calculateRandomDirectionInHemisphere(nor, rng);
    return m.color;
}

__host__ __device__
glm::vec3 sampleSpecularRefl(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {

    wi = glm::reflect(wo, nor);
    return m.specular.color;
}

__host__ __device__
glm::vec3 sampleSpecularTrans(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {


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
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // Referenced CIS561 Sample_f(): 
    // Computes the overall light scattering properties of a point on a Material,
    // given the incoming wi and outgoing wo light directions.
    // In other words, Sample_f() evaluates the BSDF *after* generating
    // a wi based on the Intersection's material properties, allowing
    // us to bias our wi samples in a way that gives more consistent
    // light scattered along wo.
    if (pathSegment.remainingBounces == 0) return;

    glm::vec3 wo = pathSegment.ray.direction;
    glm::vec3 wi = glm::vec3(0.0f);
    glm::vec3 bsdf = glm::vec3(0.0f);

    // Based on material type
    if (m.hasReflective && m.hasRefractive) {
        bsdf = sampleGlass(m, normal, rng, wo, wi);
    }
    else if (m.hasReflective) {
        bsdf = sampleSpecularRefl(m, normal, wo, wi);
    }
    else if (m.hasRefractive) {
        bsdf = sampleSpecularTrans(m, normal, wo, wi);
    }
    // default to diffuse refl
    else {
        bsdf = sampleDiffuse(m, normal, rng, wi);
    }

    pathSegment.throughput *= bsdf;
    pathSegment.ray.direction = wi;
    pathSegment.ray.origin = intersect + 0.001f * wi;
}
