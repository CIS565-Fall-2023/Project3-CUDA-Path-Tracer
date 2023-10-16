#pragma once

#include "intersections.h"
#include <cmath>
#include <thrust/random.h>
#include <glm/glm.hpp>

#define TWO_PI 6.2831853071795864769252866
#define IMPERFECT_SPECULAR 1


class BSSRDF {
typedef glm::vec3 Vec3;

public:
    float scattering_coefficient;//0.001 - 5.0
    float absorption_coefficient;//0.1 - 10.0
    float refraction_index;//1.0 - 3.0

    __host__ __device__ BSSRDF(float scattering, float absorption, float refraction)
        : scattering_coefficient(scattering), absorption_coefficient(absorption), refraction_index(refraction) {}

    __host__ __device__ float fresnel(float incident_angle) const {
        float r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - incident_angle, 5);
    }

    __host__ __device__ float phase_function(float cos_scatter_angle) const {
        float g = 0.5;
        float denominator = 1.0 + g * g - 2.0 * g * cos_scatter_angle;
        return (1.0 - g * g) / (denominator * sqrt(denominator));
    }

    __host__ __device__ float single_scattering(const Vec3& scattering_direction, const Vec3& normal) const {
        float theta = glm::dot(scattering_direction, normal);
        float adjusted_scattering = scattering_coefficient * (0.5 + 0.5 * theta);
        return adjusted_scattering;
    }

    __host__ __device__ float multiple_scattering(const Vec3& view_direction, const Vec3& new_point, const Vec3& original_point, const Vec3& light_direction) const {
        float distance = glm::length(new_point - original_point);
        float decay = exp(-absorption_coefficient * distance);

        float cos_scatter_angle = glm::dot(-view_direction, light_direction);
        float phase_factor = phase_function(cos_scatter_angle);

        float result = decay * phase_factor;
        return glm::clamp(result, 0.0f, 1.0f);
    }

    __host__ __device__
        float evaluate(const Vec3& light_direction, const Vec3& view_direction, const Vec3& point, const Vec3& original_point) const {
        float incident_angle = glm::dot(light_direction, view_direction);
        float fresnel_term = fresnel(incident_angle);
        float single_scatter = single_scattering(-view_direction, light_direction);
        float multiple_scatter = multiple_scattering(-view_direction, point, original_point, light_direction);

        // Ensure energy conservation
        float totalScatter = single_scatter + multiple_scatter;
        if (totalScatter > 1.0f) {
            single_scatter /= totalScatter;
            multiple_scatter /= totalScatter;
        }

        return fresnel_term * (single_scatter + multiple_scatter);
    }

};
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
float calculateFresnelReflectance(const glm::vec3& rayDir, glm::vec3& normal, float indexOfRefraction) {
    float cosThetaI = glm::dot(rayDir, normal);
    float R1 = indexOfRefraction;
    float R2 = 1.0f;

    if (cosThetaI < 0.0f) {
        cosThetaI = -cosThetaI;
        std::swap(R1, R2);
        normal = -normal;
    }

    //Schlick
    float R0 = pow((R1 - R2) / (R1 + R2), 2);
    return R0 + (1 - R0) * pow(1 - cosThetaI, 5);
}


__host__ __device__
float sampleHG(float g, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float random_val = u01(rng);
    if (std::abs(g) < 1e-3)
        return 2 * random_val - 1;

    float term = (1 - g * g) / (1 - g + 2 * g * random_val);
    return (1 + g * g - term * term) / (2 * g);
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
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng, 
    Geom* lights, int lightNum) {

    if (pathSegment.remainingBounces <= 0) return;

    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(0, 1);
    float random = u01(rng);
    float random2 = u02(rng);
    glm::vec3 newDir;

    if (m.hasReflective && m.hasRefractive) {
        float cosThetaI = glm::dot(pathSegment.ray.direction, normal);
        if (cosThetaI < 0.0f) {
            cosThetaI = -cosThetaI;
            normal = -normal;
        }

        float R = calculateFresnelReflectance(pathSegment.ray.direction, normal, m.indexOfRefraction);

        if (random < R) {
            newDir = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            float eta = cosThetaI > 0 ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
            newDir = glm::refract(pathSegment.ray.direction, normal, eta);
        }

        // Russian Roulette with color-based termination
        float continueProbability = glm::max(glm::max(pathSegment.color.r, pathSegment.color.g), pathSegment.color.b);
        if (u01(rng) > continueProbability) {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces--;
            return;
        }
        // Adjust color due to the probability to counteract the average reduction
        pathSegment.color /= continueProbability;
#if IMPERFECT_SPECULAR
        // Imperfect specular reflection/refraction
        newDir += calculateRandomDirectionInHemisphere(normal, rng) * 0.3f;
        newDir = glm::normalize(newDir);
#endif
        pathSegment.color *= m.specular.color * m.color;
    }
    else if (m.hasReflective) {
#if IMPERFECT_SPECULAR
        float fresnelReflectance = calculateFresnelReflectance(-pathSegment.ray.direction, normal, m.indexOfRefraction);
        if (random < fresnelReflectance)
        {
            newDir = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
        }
        else
        {
            newDir = calculateRandomDirectionInHemisphere(normal, rng);
        }
        pathSegment.color *= m.color;
#else
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color * m.color;
#endif
    }
    else if (m.hasRefractive) {
        float cosTheta = glm::dot(-pathSegment.ray.direction, normal);
        bool entering = cosTheta > 0;
        float eta = entering ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
        newDir = glm::refract(pathSegment.ray.direction, normal, eta);
        if (glm::length(newDir) < 0.01f) { // Check for total internal reflection
            newDir = glm::reflect(pathSegment.ray.direction, normal);
        }
        pathSegment.color *= m.specular.color * m.color;
}
    else if (m.hasSubsurface) {
        BSSRDF subsurfaceModel(1.2f, 1.0f, 1.3f);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float random2 = u01(rng);
        bool outside = glm::dot(pathSegment.ray.direction, normal) > 0;
        //outside = true;
        if (!outside) { // Inside the object
            float fresnelReflectance = calculateFresnelReflectance(pathSegment.ray.direction, normal, subsurfaceModel.refraction_index);
            if (random2 < fresnelReflectance) {
                glm::vec3 inverse_normal = -normal;
                glm::vec3 insideDirection = calculateRandomDirectionInHemisphere(inverse_normal, rng);
                glm::vec3 new_color = pathSegment.color * subsurfaceModel.single_scattering(insideDirection, inverse_normal) * m.color;
                pathSegment.color = new_color;
                pathSegment.ray.direction = insideDirection;
                pathSegment.ray.origin = intersect + insideDirection * 0.001f;

            }
            else {
                // Multiple Scattering
                glm::vec3 scattering_direction = calculateRandomDirectionInHemisphere(normal, rng);

                thrust::uniform_real_distribution<int> u02(0, lightNum - 1);
                Geom light = lights[u02(rng)];
                glm::vec3 lightPos = glm::vec3(light.translation * glm::vec3(u01(rng), u01(rng), u01(rng)));

                glm::vec3 new_origin = intersect + scattering_direction * m.subsurfaceRadius;
                glm::vec3 lightDir = glm::normalize(lightPos - intersect);

                glm::vec3 view_direction = glm::normalize(pathSegment.ray.origin - intersect);
                glm::vec3 new_color = pathSegment.color * subsurfaceModel.multiple_scattering(view_direction, new_origin, intersect, lightDir) * m.color;
                //glm::vec3 new_color = pathSegment.color * exp(-(intersect - pathSegment.ray.origin) * m.specular.exponent) * m.color;
                pathSegment.color = new_color;
                pathSegment.ray.origin = new_origin;
                pathSegment.ray.direction = scattering_direction;
            }

        }
        else {  // Outside the object
            glm::vec3 incomingDir = -pathSegment.ray.direction;
            float fresnelEffect = subsurfaceModel.fresnel(glm::dot(incomingDir, normal));
            glm::vec3 refractedDir = glm::refract(incomingDir, normal, 1.0f / subsurfaceModel.refraction_index);

            glm::vec3 new_color = pathSegment.color * fresnelEffect * subsurfaceModel.single_scattering(refractedDir, intersect) * m.color;
            pathSegment.color = new_color;

            glm::vec3 new_origin = intersect + glm::normalize(pathSegment.ray.direction) * 0.001f + normal;
            pathSegment.ray.origin = new_origin + refractedDir * 0.001f;
            pathSegment.ray.direction = refractedDir;
        }
        pathSegment.remainingBounces--;
        return;
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    glm::vec3 newOrigin = intersect + 0.001f * newDir;

    pathSegment.ray.origin = newOrigin;
    pathSegment.ray.direction = newDir;
    pathSegment.remainingBounces--;
    return;
}

