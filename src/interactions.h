#pragma once

#include "intersections.h"

__host__ __device__
void make_coord_space(glm::mat3x3& o2w, glm::vec3 n) {
    glm::vec3 z = glm::vec3(n.x, n.y, n.z);
    glm::vec3 h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
        h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
        h.y = 1.0;
    else
        h.z = 1.0;

    z = glm::normalize(z);
    glm::vec3 y = cross(h, z);
    y = glm::normalize(y);
    glm::vec3 x = cross(z, y);
    x = glm::normalize(x);

    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;
}

__host__ __device__
glm::vec3 hemiSphereRandomSample(thrust::default_random_engine& rng, float * pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float Xi1 = u01(rng);
    float Xi2 = u01(rng);

    float r = sqrt(Xi1);
    float phi = 2. * PI * Xi2;
    *pdf = sqrt(1 - r) * INV_PI;
    return glm::vec3(r * cos(phi), r * sin(phi), sqrt(1 - Xi1));
}

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
    if (pathSegment.remainingBounces == 0) return;
    float cosineTerm = glm::dot(-pathSegment.ray.direction, normal);
    thrust::uniform_real_distribution<float> diffuse_or_specular(0, 1);
    glm::vec3 wo;
    float prob = diffuse_or_specular(rng);
    const float splitThreshold = 0.5f;
    float NdotL = glm::dot(-pathSegment.ray.direction, normal);
    float distance = glm::distance(pathSegment.ray.origin, intersect);
    float distance2 = distance * distance;
    if (m.hasReflective || m.hasRefractive) {
        pathSegment.color *= m.specular.color;
        pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        //pathSegment.ray.direction = glm::normalize(imperfectReflection(glm::normalize(glm::reflect(pathSegment.ray.direction, normal)), rng, m.shininess));
        pathSegment.remainingBounces--;
        pathSegment.ray.origin = intersect + 0.0001f * normal;
    }

    else {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.remainingBounces--;
        pathSegment.ray.origin = intersect + 0.0001f * normal;
    }
}
