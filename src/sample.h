#pragma once

#include "utilities.h"
#include <cuda_runtime.h>

__host__ __device__
glm::vec2 concentricSampleDisk(const glm::vec2& sample)
{
    glm::vec2 uOffset = 2.f * sample - glm::vec2(1.);
    if (uOffset.x == 0 && uOffset.y == 0) return glm::vec2(0, 0);
    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI / 4.0) * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = (PI / 2.0) - (PI / 4.0) * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng, float& pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    pdf = up / PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
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
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}