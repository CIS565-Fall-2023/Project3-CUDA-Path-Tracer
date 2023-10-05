#pragma once

#include "intersections.h"
#include "utilities.h"

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
float schlickFresnel(float& cosTheta, float& eta1, float& eta2) {
    if (cosTheta < 0.0f) {
        // Ray is exiting the material, swap the indices
        std::swap(eta1, eta2);
    }

    float r0 = glm::pow((eta1 - eta2) / (eta1 + eta2), 2.0f);
    return r0 + (1.0f - r0) * glm::pow(1.0f - cosTheta, 5.0f);
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
__host__ __device__ void scatterRay(PathSegment& pathSegment, glm::vec3 intersect,
    glm::vec3 normal, const Material& m, thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 incomingDirection = pathSegment.ray.direction;
    // glm::vec3 shadeNormal = normal;
    float cosTheta = glm::clamp(glm::dot(normal, -incomingDirection), -1.0f, 1.0f);
    float eta1 = 1.0, eta2 = m.indexOfRefraction;
    float fresnelEffect = schlickFresnel(cosTheta, eta1, eta2);

    
    //if (!m.hasReflective && !m.hasRefractive) {
    //    auto direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    //    pathSegment.ray.direction = direction;
    //    pathSegment.ray.origin = intersect + 0.001f * normal;
    //    pathSegment.color *= m.color;
    //}
    //else if (m.hasRefractive && m.hasReflective) {
    //    glm::vec3 incident = pathSegment.ray.direction;
    //    glm::vec3 n = normal;
    //    float cosT = glm::dot(n, -incident);
    //    float n1, n2;
    //    if (cosT < 0.0) {
    //        n = glm::normalize(-n);
    //        n1 = m.indexOfRefraction;
    //        n2 = 1.0;
    //    }
    //    else {
    //        n1 = 1.0;
    //        n2 = m.indexOfRefraction;
    //    }

    //    float fresnel = schlickFresnel(cosT, n1, n2);
    //    thrust::uniform_real_distribution<float> u01(0, 1);
    //    if (u01(rng) < fresnel) {
    //        glm::vec3 reflect = glm::reflect(incident, n);
    //        pathSegment.ray.direction = reflect;
    //        pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
    //        pathSegment.color *= m.color;
    //    }
    //    else {
    //        glm::vec3 refract = glm::normalize(glm::refract(incident, n, n1 / n2));
    //        pathSegment.ray.direction = refract;
    //        pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
    //        pathSegment.color *= m.color;
    //    }

    //}

    float scatterChoice = u01(rng);
    float pRefract = m.hasRefractive ? fresnelEffect : 0.0f;
    float pReflect = m.hasReflective ? (1 - pRefract) * 0.5f : 0.0f;
    float pDiffuse = 1.0f - pRefract - pReflect;

    if (scatterChoice < pRefract) {
        if (cosTheta < 0) {
            normal = -normal;
        }
        
        glm::vec3 refracted = glm::refract(incomingDirection, normal, eta1 / eta2);

        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::normalize(refracted);
        pathSegment.ray.origin = intersect * STEP_SIZE * pathSegment.ray.direction;
    } 
    else if (scatterChoice < (pRefract + pReflect)) {
        if (cosTheta < 0) {
            normal = -normal;
        }
        glm::vec3 perfectReflect = glm::reflect(incomingDirection, normal);
        pathSegment.ray.direction = glm::mix(perfectReflect,
            calculateRandomDirectionInHemisphere(normal, rng),
            m.specular.exponent);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + STEP_SIZE * pathSegment.ray.direction;
    }
    else {
        glm::vec3 dir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = glm::normalize(dir);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + STEP_SIZE * pathSegment.ray.direction;
    }

    // pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
    pathSegment.remainingBounces--;
}


