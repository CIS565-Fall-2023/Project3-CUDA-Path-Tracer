#pragma once

#include "intersections.h"
#define REFLECTIVE_ONLY 0
#define DIFFUSE_ONLY 0
#define BOTH 1
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
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
    //pathSegments.color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
    glm::vec3 sampled_direction;
    thrust::uniform_real_distribution<float> u01(0, 1);
#if REFLECTIVE_ONLY
    if (m.hasReflective)
    {
        sampled_direction = glm::reflect(pathSegment.ray.direction, normal);
    }

    else
    {
#endif
#if DIFFUSE_ONLY
        sampled_direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= (m.color);
#endif
#if REFLECTIVE_ONLY
    }
#endif
#if BOTH
    // use specular.factor here instead of rng for mesh
    if (m.hasReflective && (u01(rng) > 0.07))
    {
        /*sampled_direction = u01(rng) > 0.8 ? glm::reflect(pathSegment.ray.direction, normal) 
                                           : calculateRandomDirectionInHemisphere(normal, rng);*/
        sampled_direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    }
    else
    {
		sampled_direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= (m.color);
	}
#endif
    pathSegment.ray.direction = glm::normalize(sampled_direction);
    pathSegment.ray.origin = intersect + 0.00001f * pathSegment.ray.direction;
}
