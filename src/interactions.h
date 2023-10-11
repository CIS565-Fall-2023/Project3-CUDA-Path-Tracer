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

//for refractionRation, glass is 1.3-1.7, diamond is 2.4
__host__ __device__
glm::vec3 calculateRefraction(
    float costheta, glm::vec3 normal, glm::vec3 indir, float refractionRatio)
{
    //float costheta = thrust::min(glm::dot(temp, normal), 1.0f);
    costheta = thrust::min(costheta, 1.0f);
    glm::vec3 rayOutPre = refractionRatio * (indir + costheta * normal);
    float rayOutPreSquare = rayOutPre.x * rayOutPre.x + rayOutPre.y * rayOutPre.y + rayOutPre.z * rayOutPre.z;
    glm::vec3 rayOutPara = -glm::sqrt(1 - rayOutPreSquare) * normal;
    return rayOutPre + rayOutPara;
}

__host__ __device__
glm::vec3 calculateSpecularReflection(
    glm::vec3 normal, PathSegment& pathSegment)
{
    glm::vec3 indir = pathSegment.ray.direction;
    return indir - 2.0f * glm::dot(indir, normal) * normal;
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
    Ray newRay;        

    if (m.hasRefractive)
    {
        glm::vec3 indir = pathSegment.ray.direction;
        float costheta = glm::dot(-1.0f * indir, normal);
        float sintheta = sqrtf(1 - costheta * costheta);
        float ratio = 1.0f / m.indexOfRefraction;
        //if the angle between normal and ray is more then 90 degree
        if (costheta <= 0.0f)
        {
            ratio = 1.0f / ratio;
        }
        else
        {
            --pathSegment.remainingBounces;
        }
        bool canRefract = ratio * sintheta < 1.0f;
        thrust::uniform_real_distribution<float> u01(0, 1);
        float prob = u01(rng);
        if (canRefract && prob > 0.07f)
        {
            newRay = { intersect, glm::refract(indir, normal, ratio) };
            pathSegment.color *= 1.67f * m.color;
        }
        else
        {
            newRay = { intersect, calculateSpecularReflection(normal, pathSegment) };
            pathSegment.color = pathSegment.color * m.color * 0.9f;
        }

    }
    else if (m.hasReflective)
    {
        newRay = { intersect, calculateSpecularReflection(normal, pathSegment) };
        pathSegment.color = pathSegment.color * m.color;
        --pathSegment.remainingBounces;
    }
    else
    {
        newRay = { intersect, calculateRandomDirectionInHemisphere(normal, rng) };
        pathSegment.color = pathSegment.color * m.color;
        --pathSegment.remainingBounces;
    }

    if (pathSegment.remainingBounces < 1)
    {
        pathSegment.color = glm::vec3(0.f);
        return;
    }

    pathSegment.ray = newRay;
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
    //record the current normal of the ray
    pathSegment.currentNormal = normal;
}
