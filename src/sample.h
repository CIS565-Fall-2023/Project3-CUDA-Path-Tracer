#pragma once

#include "utilities.h"
#include <cuda_runtime.h>
#include <thrust/random.h>

#define USE_STRATIFIED 0
const int RES_PER_SIDE = 100;

// ref: 561 hw
__host__ __device__
glm::vec3 squareToDiskConcentric(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 xi(u01(rng), u01(rng));

    float phi = 0.0f, r = 0.0f, u = 0.0f, v = 0.0f;
    float a = 2.0f * xi.x - 1.0f;
    float b = 2.0f * xi.y - 1.0f;

    if (a > -b)
    {
        if (a > b)
        {
            r = a;
            phi = (b / a) * (PI / 4.0f);
        }
        else
        {
            r = b;
            phi = (2.0f - (a / b)) * (PI / 4.0f);
        }
    }
    else
    {
        if (a < b)
        {
            r = -a;
            phi = (4.0f + (b / a)) * (PI / 4.0f);
        }
        else
        {
            r = -b;
            if (b != 0.0f)
            {
                phi = (6.0f - (a / b)) * (PI / 4.0f);
            }
            else
            {
                phi = 0.0f;
            }
        }
    }

    u = r * cos(phi);
    v = r * sin(phi);

    return glm::vec3(u, v, 0.0f);
}

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

#if USE_STRATIFIED
    // ref: https://fja05680.github.io/BFS_Sequences/BFS%20Sequences.pdf
    int i = u01(rng) * RES_PER_SIDE * RES_PER_SIDE;

    // (i + u) / N
    float up = sqrt(((float)(i / RES_PER_SIDE) + u01(rng)) / (float)RES_PER_SIDE); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = (((float)(i % RES_PER_SIDE)+u01(rng)) / (float)RES_PER_SIDE) * TWO_PI;

    pdf = up / PI;

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
#else
    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    pdf = up / PI;

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
#endif
}