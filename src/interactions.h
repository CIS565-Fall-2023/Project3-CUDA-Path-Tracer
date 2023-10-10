#pragma once
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <thrust/device_ptr.h>
#include <thrust/partition.h>
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
float fresnelSchlick(float cos_theta, float R0) {

    return R0 + (1 - R0) * powf(1 - cos_theta, 5.f);

}
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_num = u01(rng);

    // sub surface scatter
    if (m.hasReflective == 0.5f) {
        if (pathSegment.outside == false) {
            if (random_num < 0.5) {
                glm::vec3 inverse_normal = -normal;
                glm::vec3 insideDirection = calculateRandomDirectionInHemisphere(inverse_normal, rng);
                glm::vec3 diffuseColor = pathSegment.color *exp(-(intersect-pathSegment.ray.origin) * m.specular.exponent);
                pathSegment.color = diffuseColor;
                pathSegment.ray.direction = insideDirection;
                pathSegment.ray.origin = intersect + insideDirection * 0.001f;

            }
            else {
                float move_dist = random_num * 0.01;
                pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal,rng);
                pathSegment.ray.origin = intersect+ pathSegment.ray.direction* 0.002f;
                glm::vec3 diffuseColor = pathSegment.color * exp(-(intersect - pathSegment.ray.origin) * m.specular.exponent);
                pathSegment.color = diffuseColor;
            
            }

        }
        else {
            glm::vec3 inverse_normal = -normal;
            glm::vec3 insideDirection = calculateRandomDirectionInHemisphere(normal, rng);
            glm::vec3 new_origin = intersect + normalize(pathSegment.ray.direction) * 0.0001f + inverse_normal;

            glm::vec3 diffuseColor = pathSegment.color * m.color;
            pathSegment.color = diffuseColor;
            pathSegment.ray.origin = new_origin + insideDirection * 0.0007f ;

            pathSegment.ray.direction = insideDirection;
        }
    
    }
    else if (m.hasRefractive == 1.0f) {

        //reference: https://en.wikipedia.org/wiki/Schlick%27s_approximation
        //reference to 561hw6
        float cosTheta = glm::dot(pathSegment.ray.direction, normal);
        float f_E;

        if (cosTheta > 0) {
            f_E = m.indexOfRefraction;
        }
        else {
            f_E = 1.0f / m.indexOfRefraction;
        }
        float abs_costheta = glm::abs(cosTheta);
        float R0 = powf((1.f - f_E) / (1.f + f_E), 2.f);

        //eflection coefficient F
        float F = fresnelSchlick(abs_costheta, R0);
        glm::vec3 Wo;

        if (random_num < 0.4) {
            Wo = glm::reflect(pathSegment.ray.direction, normal);
            //Wo = calculateRandomDirectionInHemisphere(normal, rng);
        }
        else {
            Wo = glm::refract(pathSegment.ray.direction, normal, f_E);
            
        }

        pathSegment.color = pathSegment.color * m.specular.color;
        pathSegment.ray.direction = Wo;
        pathSegment.ray.origin = intersect+ Wo * 0.001f;

    }
    else if (m.hasReflective==1.0f) {
       
        //specular
        if (random_num < 0.5) {
            glm::vec3 reflectDirection = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            glm::vec3 reflectColor = pathSegment.color * m.color;

            pathSegment.color = reflectColor * 1.0f;
            pathSegment.ray.direction = reflectDirection;
            pathSegment.ray.origin = intersect;

        }
        else {
            glm::vec3 diffuseDirection = calculateRandomDirectionInHemisphere(normal, rng);
            glm::vec3 diffuseColor = pathSegment.color * m.color;
            pathSegment.color = diffuseColor * 1.0f;
            pathSegment.ray.direction = diffuseDirection;
            pathSegment.ray.origin = intersect;
        }

    }
    else {
        
        glm::vec3 diffuseDirection = calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 diffuseColor = pathSegment.color * m.color;
        pathSegment.color = diffuseColor * 1.0f;
        pathSegment.ray.direction = diffuseDirection;
        pathSegment.ray.origin = intersect;
    }
}
