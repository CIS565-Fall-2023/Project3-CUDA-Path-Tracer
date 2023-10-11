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

//---------------------------------------------------------
//---------------BSDF funcs from CIS5610 PBR---------------
//---------------------------------------------------------

__host__ __device__
glm::vec3 sampleDiffuse(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3& wi) { // write out to wi

    wi = calculateRandomDirectionInHemisphere(nor, rng);
    return m.color;
}

__host__ __device__
glm::vec3 sampleDiffuseTrans(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3& wi,
    float& absDot, float& pdf) {
    // sample wi in the opposite hemisphere of wo, everything else same as diffuse_refl
    wi = calculateRandomDirectionInHemisphere(nor, rng);
    wi.x = -wi.z;
    absDot = glm::abs(glm::dot(nor, wi));
    pdf = absDot * INV_PI;
    return m.color * INV_PI;
}

__host__ __device__
glm::vec3 sampleSpecularRefl(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {

    wi = glm::reflect(wo, nor);
    return m.specular.color;
}

__host__ __device__
glm::vec3 sampleSpecularTrans(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {

    // Default to hard-coded glass IOR if not provided
    float ior = m.indexOfRefraction;
    ior = ior ? 1.55f : ior;

    // figure out which surface to enter/exit
    // eta of air always assumed to be 1
    bool entering = glm::clamp(glm::dot(wo, nor), -1.f, 1.f) < 0.0f;
    float etaI = entering ? 1.0f : ior; // incident index
    float etaT = entering ? ior : 1.0f; // transmitted index
    float eta = etaI / etaT;

    // flip normal
    nor = entering ? nor : -nor;
    wi = glm::refract(wo, nor, eta);

    // handle total internal refl
    if (glm::length(wi) < EPSILON) {
        wi = glm::reflect(wo, nor);
        return glm::vec3(0.0f);
    }
    return m.specular.color;
}

__host__ __device__
float fresnelDielectricEval(const Material& m, float cosThetaI) {
    float etaI = 1.f;
    float etaT = m.indexOfRefraction;
    etaT = etaT < EPSILON ? 1.55f : etaT;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // see pbrt FrDielectric()
    // Potentially swap indices of refraction
    if (cosThetaI > 0.f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    cosThetaI = glm::abs(cosThetaI);

    // Computer cosThetaT using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.f,
                                         1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    
    // Handle total internal reflection
    if (sinThetaT >= 0.999f) {
        return 1.f;
    }

    // Compute Fresnel reflectance using light polarization eqns, see PBRT 8.2.1
    float cosThetaT = glm::sqrt(glm::max(0.f,
                                         1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) * 0.5f; // coefficient
}

__host__ __device__
glm::vec3 sampleGlass(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3 wo, glm::vec3& wi) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    bool random = u01(rng);

    float fresnel = fresnelDielectricEval(m, glm::dot(wo, nor));
    glm::vec3 bsdf(0.f);
    if (random < fresnel) {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        bsdf = sampleSpecularRefl(m, nor, wo, wi);
        return bsdf;
    }
    else {
        bsdf = sampleSpecularTrans(m, nor, wo, wi);
        return bsdf; // 1-fr b/c all conditions sum up to 1
    }
}

__host__ __device__
glm::vec3 samplePlastic(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3 wo, glm::vec3& wi,
    float& absDot, float& pdf) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    bool random = u01(rng);
    glm::vec3 bsdf(0.f);

    float fresnel = fresnelDielectricEval(m, glm::dot(wo, nor));;
    //if (random < fresnel) {
    //    // Have to double contribution b/c we only sample
    //    // reflection BxDF half the time
    //    bsdf = sampleSpecularRefl(m, nor, wo, wi);
    //    return bsdf;
    //}
    //else {
    //    bsdf = sampleDiffuse(m, nor, rng, wi);
    //    return bsdf; // 1-fr b/c all conditions sum up to 1
    //}

    if (random < 0.5f) {
        // diffuse
        bsdf = sampleDiffuse(m, nor, rng, wi);// glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        absDot = glm::abs(glm::dot(nor, wi));
        pdf = absDot * INV_PI;
        bsdf = m.color * INV_PI;
        bsdf *= 1.f - fresnel;
    }
    else {
        // spec refl
        bsdf = sampleSpecularRefl(m, nor, wo, wi);// glm::reflect(wo, normal);
        absDot = glm::abs(glm::dot(nor, wi));
        pdf = 1.0f;
        if (absDot > 0.0f) {
            bsdf = m.specular.color / absDot;
        }
        bsdf *= fresnel;
    }
    return bsdf * 2.0f;
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
    // Referenced CIS561 Sample_f(): 
    // Computes the overall light scattering properties of a point on a Material,
    // given the incoming wi and outgoing wo light directions.
    // In other words, Sample_f() evaluates the BSDF *after* generating
    // a wi based on the Intersection's material properties, allowing
    // us to bias our wi samples in a way that gives more consistent
    // light scattered along wo.
    if (pathSegment.remainingBounces == 0) return;

    glm::vec3 wo = pathSegment.ray.direction;
    glm::vec3 wi(0.0f);
    glm::vec3 bsdf(0.0f);
    float absDot = 1.f, pdf = 1.f;
    bool isDiffuse = false;
    bool manualCalc = false;

    // Based on material type
    if (m.hasReflective && m.hasRefractive) {
        bsdf = sampleGlass(m, normal, rng, wo, wi);
    }
    // Had to manually calculate absDot and pdf to get correct results, otherwise not working properly
    else if (m.hasReflective && glm::length(m.color) > EPSILON) {
        bsdf = samplePlastic(m, normal, rng, wo, wi, absDot, pdf);
        manualCalc = true;
    }
    else if (m.hasReflective) {
        bsdf = sampleSpecularRefl(m, normal, wo, wi);
    }
    else if (m.hasRefractive) {
        bsdf = sampleSpecularTrans(m, normal, wo, wi);
    }
    // not working properly?
    else if (m.hasTransmission) {
        bsdf = sampleDiffuseTrans(m, normal, rng, wi, absDot, pdf);
        isDiffuse = true;
        manualCalc = true;
    }
    else { // default to lambert diffuse
        bsdf = sampleDiffuse(m, normal, rng, wi);
        isDiffuse = true;
    }

    pathSegment.throughput *= manualCalc ? (bsdf * absDot / pdf) : bsdf;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.ray.origin = isDiffuse ? intersect :
        intersect + OFFSET * pathSegment.ray.direction;
    pathSegment.remainingBounces--;
}
