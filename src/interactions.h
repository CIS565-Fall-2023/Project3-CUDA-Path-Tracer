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


__host__ __device__ float calcFrensel(const PathSegment& pathSegment, glm::vec3 normal, const Material& m) {
    // cosine angle between incident ray and normal
    float cosTheta = glm::dot(pathSegment.ray.direction, -normal);

    // swap index if inside
    float ior_i = 1.0f;
    float ior_o = m.indexOfRefraction;

    // calculate reflection coefficient using Schlick's approx.
    float r0 = powf((ior_i - ior_o) / (ior_i + ior_o), 2.0f);
    return r0 + (1.0f - r0) * powf((1.0f - cosTheta), 5.0f);
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
        bool outside,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.\

    // no more bounce
    if (pathSegment.remainingBounces == 0) return;

    // if intersects light source
    if (m.emittance > 0.0f) {
        pathSegment.color *= m.color * m.emittance;
        pathSegment.remainingBounces = 0;
        return;
    }

    pathSegment.remainingBounces--;

    //if (m.sssDepth > 0) {
    //    thrust::uniform_real_distribution<float> u01(0, 1);
    //    float scatterDistance = -m.sssDepth * log(1 - u01(rng));
    //    //glm::vec3 samplePoint = intersect + scatterDistance * pathSegment.ray.direction;

    //    glm::vec3 randomDirectionInMaterial = calculateRandomDirectionInHemisphere(-pathSegment.ray.direction, rng);
    //    glm::vec3 samplePoint = intersect + scatterDistance * randomDirectionInMaterial;
    //    glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);

    //    pathSegment.ray.origin = samplePoint + 0.001f * newDirection;
    //    pathSegment.ray.direction = newDirection;
    //    pathSegment.color *= m.sssAlbedo; 
    //    return;
    //}
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_num = u01(rng);

    if (glm::length(m.sigma_a) > 0) {
       /* pathSegment.color = glm::vec3(0);
        return;*/
        glm::vec3 accumulatedColor(1.0f);

        for (int i = 0; i < 10; ++i) {
            if (outside) {
                glm::vec3 insideDirection = calculateRandomDirectionInHemisphere(-normal, rng);
                pathSegment.ray.direction = insideDirection;
                pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
                outside = false;
            }
            else {
                glm::vec3 insideDirection = calculateRandomDirectionInHemisphere(pathSegment.ray.direction, rng);

                // Attenuation based on path length
                float dist = glm::length(intersect - pathSegment.ray.origin);
                glm::vec3 transmittance = exp(-dist * m.sigma_a);

                accumulatedColor *= transmittance;

                // Check for ray exit based on scattering coefficient
                float probabilityOfScattering = glm::length(m.sigma_s) / (glm::length(m.sigma_s) + glm::length(m.sigma_a));
                if (random_num > probabilityOfScattering) {
                    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(pathSegment.ray.direction, rng);
                    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
                    outside = true;
                    break; // Exit the loop if the ray leaves the material
                }
                else {
                    pathSegment.ray.direction = insideDirection;
                    pathSegment.ray.origin = intersect + insideDirection * 0.001f;
                }
            }
        }

        pathSegment.color *= accumulatedColor;
        return;
    }
    
    if (m.hasRefractive) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float r = u01(rng); // generate a random number.
        float cosTheta = - glm::dot(normal, pathSegment.ray.direction);
        float ior = m.indexOfRefraction;
        float r_0 = ((1 - ior) / (1 + ior)) * ((1 - ior) / (1 + ior));
        float fTheta = r_0 + (1 - r_0) * powf((1 - cosTheta), 5);
        if (r < fTheta) {
            // reflect
            pathSegment.ray.origin = intersect + 0.0001f * pathSegment.ray.direction;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
        }
        else {
            // refract
            float eta = outside ? 1 / ior : ior;
            // ?? why 0.001 instead of 0.0001
            pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, eta);
            pathSegment.color *= m.color;
        }
        return;
    }
    
    if (m.hasReflective) {
        // pure specular reflection
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else {
        // diffusion
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);   
    }
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + 0.0001f * pathSegment.ray.direction;
}
