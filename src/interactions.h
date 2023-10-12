#pragma once

#include "intersections.h"

#define M_PI         3.14159265358979323846f
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
glm::vec3 calculateRandomDirectionInHemisphereStratified(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    int sqrtVal = 20;
    float invSqrtVal = 1.f / sqrtVal;
    int numSamples = sqrtVal * sqrtVal;

    int i = glm::min((int)(u01(rng) * numSamples), numSamples - 1);
    int y = i / sqrtVal;
    int x = i % sqrtVal;

    float xx = (x + u01(rng) - 0.5) * invSqrtVal;
    float yy = (y + u01(rng) - 0.5) * invSqrtVal;

    float up = sqrt(xx); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = yy * TWO_PI;

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
glm::vec3 FresnelDielectricEval(float cosThetaI) {
    // We will hard-code the indices of refraction to be
    // those of glass
    float etaI = 1.;
    float etaT = 1.55;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // TODO: Fill in the rest
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = abs(cosThetaI);
    }
    float sinThetaI = glm::sqrt(glm::max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    float cosThetaT = glm::sqrt(glm::max(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    glm::vec3 ret = glm::vec3(Rparl * Rparl + Rperp * Rperp) * 0.5f;
    return ret;
}

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

    thrust::uniform_real_distribution<float> u01(0, 1);
    float sample = u01(rng);
    glm::vec3 wi;
    
    if (m.type == SPEC_REFL) {    // Perfect specular
        wi = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.throughput *= m.specular.color;
    } 
    else if (m.type == SPEC_TRANS) {  // Only refraction
        float etaA = 1.;
        float etaB = m.indexOfRefraction;
        float entering = glm::dot(normal, pathSegment.ray.direction) < 0;
        float eta = entering ? etaA / etaB : etaB / etaA;
        glm::vec3 n = entering ? normal : -normal;
        
        wi = glm::refract(pathSegment.ray.direction, n, eta);
        if (length(wi) >= 0.) {
            pathSegment.throughput *= m.color;
        }
        else {
            pathSegment.throughput *= 0.f;
        }     
    }
    else if (m.type == SPEC_GLASS) { //Refraction & specular
        float random = u01(rng);
        if (random < 0.5) {
            wi = glm::reflect(pathSegment.ray.direction, normal);
            glm::vec3 fresn = 2.f * FresnelDielectricEval(dot(normal, normalize(wi)));
            pathSegment.throughput *= m.specular.color * fresn;
        }
        else {       
            float etaA = 1.;
            float etaB = m.indexOfRefraction;
            float entering = glm::dot(normal, pathSegment.ray.direction) < 0;
            float eta = entering ? etaA / etaB : etaB / etaA;
            glm::vec3 n = entering ? normal : -normal;

            wi = glm::refract(pathSegment.ray.direction, n, eta);
            if (length(wi) > 0.) {
                glm::vec3 fresn = 2.f * (glm::vec3(1.) - FresnelDielectricEval(dot(normal, normalize(wi))));
                pathSegment.throughput *= m.color * fresn;
            }       
        }
    }
    else if (m.type == PLASTIC) {   //Diffuse & specular
        float random = u01(rng);
        if (random < 0.5) {
            wi = glm::reflect(pathSegment.ray.direction, normal);
            glm::vec3 fresn = 2.f * FresnelDielectricEval(dot(normal, normalize(wi)));
            pathSegment.throughput *= m.specular.color * fresn;
        }
        else {
            wi = glm::normalize(calculateRandomDirectionInHemisphereStratified(normal, rng));
            glm::vec3 fresn = 2.f * (glm::vec3(1.) - FresnelDielectricEval(dot(normal, normalize(wi))));
            pathSegment.throughput *= m.color * fresn;
        }
    }
    else if (m.type == DIFFUSE) {
        wi = glm::normalize(calculateRandomDirectionInHemisphereStratified(normal, rng));
        pathSegment.throughput *= m.color;   
    }

    pathSegment.ray.direction = wi;
    pathSegment.ray.origin = intersect + (wi * 0.001f);
    
}

