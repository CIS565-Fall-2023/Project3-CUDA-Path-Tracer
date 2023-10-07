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

// The Fresnel equations describe the amount of light reflected from a surface;
__host__ __device__
float schlickFresnel(float cosTheta, float eta1, float eta2) {
    float r0 = glm::pow((eta1 - eta2) / (eta1 + eta2), 2.0f);
    return r0 + (1.0f - r0) * glm::pow(1.0f - cosTheta, 5.0f);
}

__host__ __device__ void scatterRay(PathSegment& pathSegment, glm::vec3 intersect,
    glm::vec3 normal, const Material& m, thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 incomingDir = pathSegment.ray.direction;
    glm::vec3 n = normal;

    float cosT = glm::dot(n, -incomingDir);
    float n1, n2;
    bool insideObject = false;
    if (cosT < 0.0) {
        n = glm::normalize(-n);
        n1 = m.indexOfRefraction;
        n2 = 1.0;
        insideObject = true;
    }
    else {
        n1 = 1.0;
        n2 = m.indexOfRefraction;
    }

    // Sheen influence - affecting the reflection at grazing angles
    if (m.sheen > 0.0f && cosT < 0.2f) {
        float sheenAmount = m.sheen * (1.0f - cosT) / 0.2f;
        pathSegment.color = glm::mix(pathSegment.color, glm::vec3(1.0f), sheenAmount);
    }

    if (insideObject) {
        // Apply absorption
        float distance = glm::length(intersect - pathSegment.ray.origin);
        pathSegment.color *= 0.f;// glm::exp(-m.transmittance * distance);
    }

    float fresnel = schlickFresnel(cosT, 1.0, m.indexOfRefraction);
    fresnel = glm::mix(fresnel, m.metallic, m.metallic);                         // Consider metallic property
    glm::vec3 randomDir = calculateRandomDirectionInHemisphere(n, rng);
    
    if (m.hasRefractive && m.hasReflective) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < fresnel) {
            glm::vec3 perfectReflectDir = glm::reflect(incomingDir, n);
            pathSegment.ray.direction = glm::normalize(glm::mix(perfectReflectDir, randomDir, m.roughness * m.roughness));
            pathSegment.color *= m.specular.color;
        }
        else {
            glm::vec3 perfectRefraction = glm::refract(incomingDir, n, n1 / n2);
            pathSegment.ray.direction = glm::normalize(glm::mix(perfectRefraction, randomDir, m.roughness * m.roughness));
            pathSegment.color *= m.diffuse;
        }
    }
    else if (m.hasReflective) {
        glm::vec3 perfectReflectDir = glm::reflect(incomingDir, n);
        pathSegment.ray.direction = glm::normalize(glm::mix(perfectReflectDir, randomDir, m.roughness * m.roughness));
        pathSegment.color *= m.specular.color;
    }
    else {
        pathSegment.ray.direction = glm::normalize(randomDir);
        pathSegment.color *= m.diffuse * (1.0f - m.metallic);
    }

    pathSegment.ray.origin = intersect + OFFSET * pathSegment.ray.direction;
    pathSegment.remainingBounces--;
}
