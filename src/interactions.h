#pragma once

#include "intersections.h"

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
        glm::vec3 normal, thrust::default_random_engine &rng, float& pdf) {
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

__host__ __device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (g * g + f * f);
}

__host__ __device__ float fresnelDielectricEval(float cosi, float etai, float etat) {
    if (cosi > 0.0f) {
        float tmp = etai;
        etai = etat;
        etat = tmp;
    }

    cosi = abs(cosi);

    float sint = (etai / etat) * sqrtf(fmaxf(0.0f, 1.0f - cosi * cosi));
    if (sint >= 1.0f) {
        // total internal reflection
        return 1.0;
    }
    
    float cost = sqrtf(fmaxf(0.0f, 1.0f - sint * sint));

    float Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    float Re = (Rparl * Rparl + Rperp * Rperp) / 2.0f;

    return Re;
}

/*Sample material color*/
__host__ __device__ glm::vec3 sampleSpecularReflectMaterial(
    const Material& m, const glm::vec3& normal, const glm::vec3& wo, glm::vec3& wi) {
    wi = glm::normalize(glm::reflect(wo, normal));
    return m.specular.color;
}

__host__ __device__ glm::vec3 sampleSpecularTransimissionMaterial(
    const Material& m, const glm::vec3& normal, const glm::vec3& wo, glm::vec3& wi) {
    float etaA = 1.0f, etaB = m.indexOfRefraction;
    float ni = dot(wo, normal);
    glm::vec3 nor = normal;
    bool entering = ni < 0;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    if (!entering) nor = -nor;

    // total internal reflection
    wi = etaI / etaT * sqrtf(fmaxf(0.0f, 1.0f - ni * ni)) > 1.0f ?
        glm::normalize(glm::reflect(wo, nor)) :
        glm::normalize(glm::refract(wo, nor, etaI / etaT));
    return m.specular.color;
}

__host__ __device__ glm::vec3 sampleFresnelSpecularMaterial(
    const Material& m, const glm::vec3& normal, 
    const glm::vec3& wo, glm::vec3& wi, thrust::default_random_engine& rng) {

    float cosThetaI = glm::abs(glm::dot(wo, normal));
    float sinThetaI = std::sqrt(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float F = fresnelDielectricEval(cosThetaI, 1.0f, m.indexOfRefraction);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float u = u01(rng);
    if (u < F) {
        // reflect
        return F * sampleSpecularReflectMaterial(m, normal, wo, wi);
    } else {
        // transmission
        return (1.0f - F) * sampleSpecularTransimissionMaterial(m, normal, wo, wi);
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
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        glm::vec3 tangent,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng); // TODO: rand ratio between diffuse and specular

    float pdf = 1.0;
    glm::vec3 dir = pathSegment.ray.direction;

    if (m.hasReflective > 0.0 && m.hasRefractive > 0.0) 
    {
        pathSegment.color *= sampleFresnelSpecularMaterial(m, normal, dir, pathSegment.ray.direction, rng);
    } 
    else if (m.hasReflective > 0.0) 
    {
        // perfect specular
        pathSegment.color *= sampleSpecularReflectMaterial(m, normal, dir, pathSegment.ray.direction);

        // imperfect
        //float x1 = u01(rng), x2 = u01(rng);
        //float theta = acos(pow(x1, 1.0 / (m.specular.exponent + 1)));
        //float phi = 2 * PI * x2;

        //glm::vec3 s = glm::vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

        //// sample direction must be transformed to world space
        //// tangent-space to world-space: local tangent, binormal, and normal at the surface point
        //glm::vec3 binormal = glm::normalize(glm::cross(normal, tangent));
        //glm::mat3 TBN = glm::mat3(tangent, binormal, normal);

        //glm::vec3 r = glm::normalize(glm::transpose(TBN) * pathSegment.ray.direction); // world-space to tangent-space

        //// specular-space to tangent-space
        //glm::mat3 sampleToTangent;
        //sampleToTangent[2] = r;
        //sampleToTangent[1] = glm::normalize(glm::vec3(-r.y, r.x, 0.0f));
        //sampleToTangent[0] = glm::normalize(glm::cross(sampleToTangent[1], sampleToTangent[2]));

        //// specular-space to world-space
        //glm::mat3 mat = TBN * sampleToTangent;

        //pathSegment.ray.direction = mat * s;
    }
    else if (m.hasRefractive > 0.0)
    {
        pathSegment.color *= sampleSpecularTransimissionMaterial(m, normal, dir, pathSegment.ray.direction);
    } 
    else
    {
        // diffuse
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng, pdf));
        pathSegment.color *= (m.color *abs(glm::dot(glm::normalize(pathSegment.ray.direction), normal)) / PI) / pdf;
    }

    --pathSegment.remainingBounces;
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
}


__host__ __device__
glm::vec3 sampleLight(
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3& wi,
    int& chosenLightIndex,
    thrust::default_random_engine& rng,
    const Light* lights,
    const int& num_lights) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    chosenLightIndex = (int)(u01(rng) * num_lights);

    Light light = lights[chosenLightIndex];
    switch (light.lightType)
    {
        
    }

    return glm::vec3(0.0f);
}