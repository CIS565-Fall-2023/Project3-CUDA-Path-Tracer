#pragma once

#include "intersections.h"
#include <thrust/random.h>

__host__ __device__
glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
    // adapted from PBRT 13.6.2
    glm::vec2 uOffset = 2.f * u - 1.f;
    if (uOffset == glm::vec2(0.f)) {
        return uOffset;
    }
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = QUARTER_PI * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = HALF_PI - QUARTER_PI * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(cos(theta), sin(theta));
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

__host__ __device__
void Sample_f_diffuse(glm::vec3 color, thrust::default_random_engine& rng,
    glm::vec3 intersect, glm::vec3 normal, PathSegment& pathSegment) {
    glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
    float pdf = wi.z * INV_PI;
    if (pdf == 0.f) {
        // terminate on invalid sample
        pathSegment.remainingBounces = 0;
        pathSegment.color = glm::vec3(0.f);
    }
    else {
        pathSegment.color *= color; // * INV_PI * glm::abs(glm::dot(wi, normal)) / (glm::abs(glm::dot(wi, normal)) * INV_PI) the other terms cancel out
        pathSegment.ray.origin = intersect + 0.0001f * wi;
        pathSegment.ray.direction = wi;
    }
}

__host__ __device__
void Sample_f_specular_refl(glm::vec3 color, glm::vec3 intersect, glm::vec3 normal,
    PathSegment& pathSegment) {
    glm::vec3 wi = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= color; // pdf = 1, abscos(theta_wo) and absdot(wi, n) = cos(theta_j) cancels out
    pathSegment.ray.origin = intersect + 0.0001f * wi;
    pathSegment.ray.direction = wi;
}

__host__ __device__
void Sample_f_specular_trans(glm::vec3 color, glm::vec3 intersect, glm::vec3 normal,
    float ior, PathSegment& pathSegment) {
    glm::vec3 wi;
    if (glm::dot(pathSegment.ray.direction, normal) < 0) {
        wi = glm::normalize(glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(normal), 1.f / ior));
    } else {
        wi = glm::normalize(glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(-normal), ior));
    }
    pathSegment.color *= color; // pdf = 1, abscos(theta_wo) and absdot(wi, n) = cos(theta_j) cancels out
    pathSegment.ray.origin = intersect + 0.0001f * wi;
    pathSegment.ray.direction = wi;
}

__host__ __device__
glm::vec3 FresnelDielectricEval(float cosThetaI, float ior) {
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    float etaI = 1.f;
    float etaT = ior;
    
    if (cosThetaI <= 0.f) {
        etaI = ior;
        etaT = 1.f;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(FLOATMAX(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.f) {
        return glm::vec3(1.f);
    }
    float cosThetaT = sqrt(FLOATMAX(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2);
}

__host__ __device__
void Sample_f_specular_fresnel(glm::vec3 color, thrust::default_random_engine& rng,
    glm::vec3 intersect, glm::vec3 normal, float ior, PathSegment& pathSegment) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    // double contribute since only sample each half the times
    if (u01(rng) < 0.5) {
        Sample_f_specular_refl(color, intersect, normal, pathSegment);
        pathSegment.color *= 2.f * FresnelDielectricEval(glm::dot(pathSegment.ray.direction, normal), ior);
    } else {
        Sample_f_specular_trans(color, intersect, normal, ior, pathSegment);
        pathSegment.color *= 2.f * (glm::vec3(1.f) - FresnelDielectricEval(glm::dot(pathSegment.ray.direction, normal), ior));
    }
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
__device__
void scatterRay(
        PathSegment &pathSegment,
        glm::vec3 intersect,
        ShadeableIntersection &isect,
        const Material &m,
        const cudaTextureObject_t* const texObjs,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // sample f
    // if pdf invalid, set numBounces = 0, set color black
    // multiply color by throughput
    // spawn new ray
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.textureid >= 0) {
        float4 texCol = tex2D<float4>(texObjs[m.textureid], isect.uv.x, isect.uv.y);
        glm::vec4 col = glm::vec4(texCol.x, texCol.y, texCol.z, texCol.w);
        if (u01(rng) < col.a) {
            Sample_f_diffuse(glm::vec3(col), rng, intersect, isect.surfaceNormal, pathSegment);
            pathSegment.color /= fmax(col.a, EPSILON);
        } else {
            pathSegment.ray.origin = intersect + 0.0001f * pathSegment.ray.direction;
            pathSegment.color /= fmax(1.f - col.a, EPSILON);
        }
        // Sample_f_diffuse(isect.surfaceNormal, rng, intersect, isect.surfaceNormal, pathSegment);
        return;
    }
    if (m.color == glm::vec3(0.f) && m.specular.color == glm::vec3(0.f)) {
        // black always gives black color in our sampling methods
        return;
    }
    if (!(m.hasReflective || m.hasRefractive)) {
       /* float4 texCol = tex2D<float4>(texObjs[15], u01(rng), u01(rng));
        glm::vec3 col = glm::vec3(texCol.x, texCol.y, texCol.z);*/
        Sample_f_diffuse(m.color, rng, intersect, isect.surfaceNormal, pathSegment);
        return;
    }
    
    float diffuseLumin = LUMINANCE(m.color);
    float specularLumin = LUMINANCE(m.specular.color);
    float pSampleDiffuse = diffuseLumin / (diffuseLumin + specularLumin);
    if (u01(rng) < pSampleDiffuse) {
        Sample_f_diffuse(m.color, rng, intersect, isect.surfaceNormal, pathSegment);
        pathSegment.color /= fmax(pSampleDiffuse, EPSILON);
    } else {
        if (!m.hasReflective) {
            Sample_f_specular_trans(m.specular.color, intersect, isect.surfaceNormal, m.indexOfRefraction, pathSegment);
        } else if (!m.hasRefractive) {
            Sample_f_specular_refl(m.specular.color, intersect, isect.surfaceNormal, pathSegment);
        } else {
            Sample_f_specular_fresnel(m.specular.color, rng, intersect, isect.surfaceNormal, m.indexOfRefraction, pathSegment);
        }
        if (glm::dot(pathSegment.ray.direction, pathSegment.ray.direction) == 0) {
            pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
            pathSegment.remainingBounces = 0;
        } else {
            pathSegment.color /= fmax((1 - pSampleDiffuse), EPSILON);
        }
    }
}
