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

/// <summary>
/// Handles lambertian diffuse reflection
/// </summary>
/// <param name="wi"></param>
/// <param name="pathSegment"></param>
/// <param name="normal"></param>
/// <param name="mat"></param>
/// <param name="diffusePDF"></param>
/// <param name="rng"></param>
/// <returns></returns>
__host__ __device__ glm::vec3 sample_f_diffuse(glm::vec3 &wi,
                                            PathSegment &pathSegment,
                                            const glm::vec3 &normal, 
                                            const Material &mat,
                                            thrust::default_random_engine& rng)
{
    wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    return mat.color;
}

/// <summary>
/// Handles specular reflection
/// </summary>
/// <param name="wi"></param>
/// <param name="pathSegment"></param>
/// <param name="normal"></param>
/// <param name="mat"></param>
/// <param name="specularPDF"></param>
/// <returns></returns>
__host__ __device__ glm::vec3 sample_f_specular_reflection(glm::vec3& wi,
                                            PathSegment& pathSegment,
                                            const glm::vec3& normal,
                                            const Material& mat)
{
    wi = glm::reflect(pathSegment.ray.direction, normal);
    return mat.specular.color;
}

/// <summary>
/// Handles refraction and total internal reflection
/// </summary>
/// <param name="wi"></param>
/// <param name="pathSegment"></param>
/// <param name="normal"></param>
/// <param name="mat"></param>
/// <param name="specularPDF"></param>
/// <returns></returns>
__host__ __device__ glm::vec3 sample_f_specular_transmission(glm::vec3& wi,
                                            PathSegment& pathSegment,
                                            glm::vec3& normal,
                                            const Material& mat)
{
    // Hard-coded to index of refraction of air
    float etaA = 1.f;
    float etaB = mat.indexOfRefraction; // IOR of material

    float eta;
    if (glm::dot(pathSegment.ray.direction, normal) < 0.0f)  // cos theta b/w normal and outgoing ray wo. dir and normal are normalized so we can simply dot it
    {
        // entering the surface!
        eta = etaA / etaB;
    }
    else
    {
        // exiting the surface!
        eta = etaB / etaA;
        normal *= -1.0f;        // flip the normal so its pointing INSIDE the surface in this case
    }

    wi = glm::refract(pathSegment.ray.direction, normal, eta);  // refracted ray
    if (glm::dot(wi, wi) > 0.0f)    // just check squared length
    {
        // ray was successfully refracted
        // don't do anything in this case
    }
    else
    {
        // total internal reflection happened!
        wi = glm::reflect(pathSegment.ray.direction, normal);
    }

    return mat.specular.color;
}

/// <summary>
/// Computes the Fresnel reflection coefficient at a given point of intersection on a surface.
/// This is for more physically based materials like glass than simply "just reflective" or "just transmissive" materials.
/// Based on PBRT and my homework from CIS 561
/// </summary>
/// <param name="cosThetaI"></param>
/// <param name="IOR"></param>
/// <returns></returns>
__host__ __device__ glm::vec3 fresnelDielectricEval(float cosThetaI, float IOR)
{
    float etaI = 1.0f;      // hard coded to IOR of air
    float etaT = IOR;
    if (cosThetaI < 0.0f)
    {
        float tmp = etaT;
        etaT = etaI;
        etaI = tmp;
        cosThetaI = std::abs(cosThetaI);
    }

    cosThetaI = max(min(cosThetaI, -1.0f), 1.0f);

    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return glm::vec3(Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

__host__ __device__ glm::vec3 sample_f_glass(glm::vec3& wi,
                                        PathSegment& pathSegment,
                                        glm::vec3& normal,
                                        const Material& mat,
                                        thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);
    glm::vec3 bsdf(1.0f);
    glm::vec3 wo = pathSegment.ray.direction;
    if (random < 0.5f) {
        // Have to double contribution b/c we only sample
        // reflection BRDF half the time
        bsdf = sample_f_specular_reflection(wi, pathSegment, normal, mat);
        return 2.0f * fresnelDielectricEval(glm::dot(normal, wi), mat.indexOfRefraction) * bsdf;
    }
    else {
        // Have to double contribution b/c we only sample
        // transmit BTDF half the time
        bsdf = sample_f_specular_transmission(wi, pathSegment, normal, mat);
        return 2.0f * (glm::vec3(1.) - fresnelDielectricEval(dot(normal, wi), mat.indexOfRefraction)) * bsdf;
    }
}

__host__ __device__ glm::vec3 sample_f_plastic(glm::vec3& wi,
                                    PathSegment& pathSegment,
                                    glm::vec3& normal,
                                    const Material& mat,
                                    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);
    glm::vec3 bsdf(1.0f);
    if (random < 0.5f) {
        // Have to double contribution b/c we only sample
        // transmit BRDF half the time
        bsdf = sample_f_specular_reflection(wi, pathSegment, normal, mat);
        return 2.0f * (fresnelDielectricEval(dot(normal, wi), mat.indexOfRefraction)) * bsdf;
    }
    else {
        // Have to double contribution b/c we only sample
        // reflection BSDF half the time
        bsdf = sample_f_diffuse(wi, pathSegment, normal, mat, rng);
        return 2.0f * bsdf;
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

__host__ __device__ void sample_f(PathSegment &pathSegment,
                                  const glm::vec3 &intersectionPoint,
                                  glm::vec3 &normal,
                                  const Material &mat,
                                  thrust::default_random_engine &rng) {
    // Renaming this from scatterRay() to sample_f() 
    // because we're sampling a new "bounce" (new wi) and also computing bxdf at the intersection point

    glm::vec3 wi(0.0f); // next iteration ray

    thrust::uniform_real_distribution<float> u01(0, 1);

    float probability = u01(rng);
    glm::vec3 bsdf(1.0f);
    float pdf = 1.0f;

    if (mat.hasReflective && mat.hasRefractive)
    {
        // glass / water, etc. transparent
        bsdf = sample_f_glass(wi, pathSegment, normal, mat, rng);
        pdf = 1.0f; // we're taking care of pdf in the function itself so anything other than 1.0f here will not be correct
    }
    else if (mat.hasReflective && glm::length(mat.color) > 0.0f)
    {
        // lambertian + specular, basically plastic material
        bsdf = sample_f_plastic(wi, pathSegment, normal, mat, rng);
        pdf = 1.0f; // again, taking care of pdf in the function itself
    }
    else if (mat.hasReflective)
    {
        // reflective only, no refraction or diffuse
        bsdf = sample_f_specular_reflection(wi, pathSegment, normal, mat);
        pdf = 1.0f;     // technically PDF for specular reflections is a delta distribution
                        // and the "usable" ray that is a perfect reflection
                        // will always have a PDF of 1 since that's the only ray we can use
    }
    else if (mat.hasRefractive)
    {
        // refraction only, no reflection or diffuse
        bsdf = sample_f_specular_transmission(wi, pathSegment, normal, mat);
        pdf = 1.0f;     // pdf is same as above
    }
    else
    {
        // diffuse
        bsdf = sample_f_diffuse(wi, pathSegment, normal, mat, rng);
        pdf = 1.0f;     // pdf is same as above
                        // in pure lambertian reflection the correct PDF would be cosTheta / pi
                        // since light is scattered uniformly in all directions
                        // it would then, however, get cancelled by the albedo term which should really be "albedo/pi"
                        // and the lambert's law term in the rendering equation. 
                        // Hence keeping this at 1.0f
    }

    // move the origin a bit in the direction of wi
    pathSegment.ray.origin = intersectionPoint + wi * 0.01f;
    pathSegment.accum_throughput *= bsdf / pdf;     // the division by pdf right now is pretty useless,
                                                    // but will come in handy if and when I
                                                    // implement more physically accurate materials
                                                    // like microfacets. Each "sample_f_x" function
                                                    // handles PDFs on its own right now
    pathSegment.ray.direction = wi;
    pathSegment.remainingBounces--;
}
