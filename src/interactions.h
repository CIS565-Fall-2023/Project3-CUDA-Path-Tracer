#pragma once

#include "intersections.h"
#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng, float& z) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
    z = up;

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
glm::vec3 Sample_wh(
    glm::vec3 wo, thrust::default_random_engine& rng, float roughness
) {
    glm::vec3 wh;
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng);
    float cosTheta = 0;
    float phi = TWO_PI * u01(rng);
    float tanTheta2 = roughness * roughness * xi / (1.0f - xi);
    cosTheta = 1.0f / sqrt(1.0f + tanTheta2);

    float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    if (wo.z * wh.z <= 0) {
        wh = -wh;
    }
    return wh;
}


__host__ __device__
float CosTheta(glm::vec3 w) { return w.z; }

__host__ __device__
float Cos2Theta(glm::vec3 w) { return w.z * w.z; }

__host__ __device__
float AbsCosTheta(glm::vec3 w) { return abs(w.z); }

__host__ __device__
float Sin2Theta(glm::vec3 w) {
    return max(0.f, 1.f - Cos2Theta(w));
}

__host__ __device__
float SinTheta(glm::vec3 w) { return sqrt(Sin2Theta(w)); }

__host__ __device__
float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }


__host__ __device__
float Tan2Theta(glm::vec3 w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

__host__ __device__
float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

__host__ __device__
float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

__host__ __device__
float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }

__host__ __device__
float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }

__host__ __device__
float Lambda(glm::vec3 w, float roughness) {
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;

    // Compute alpha for direction w
    float alpha =
        sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__host__ __device__
float TrowbridgeReitzD(glm::vec3 wh, float roughness) {
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;

    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

    float e =
        (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) *
        tan2Theta;
    // float e = (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) * tan2Theta / (roughness * roughness));
    return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
}

__host__ __device__
float TrowbridgeReitzG(glm::vec3 wo, glm::vec3 wi, float roughness) {
    return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
}

__host__ __device__
float TrowbridgeReitzPdf(glm::vec3 wo, glm::vec3 wh, float roughness) {
    return TrowbridgeReitzD(wh, roughness) * abs(wh.z);
}


__host__ __device__
glm::vec3 f_microfacet_refl(glm::vec3 albedo, glm::vec3 wo, glm::vec3 wi, float roughness) {
    float cosThetaO = AbsCosTheta(wo);
    float cosThetaI = AbsCosTheta(wi);
    glm::vec3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return glm::vec3(0.f);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return glm::vec3(0.f);
    wh = normalize(wh);
    // TODO: Handle different Fresnel coefficients
    glm::vec3 F = glm::vec3(1.);//fresnel->Evaluate(glm::dot(wi, wh));
    float D = TrowbridgeReitzD(wh, roughness);
    float G = TrowbridgeReitzG(wo, wi, roughness);
    return albedo * D * G * F /
        (4 * cosThetaI * cosThetaO);
}

__host__ __device__
void coordinateSystem(glm::vec3& v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

__host__ __device__
glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__host__ __device__
glm::mat3 WorldToLocal(glm::vec3 nor) {
    return transpose(LocalToWorld(nor));
}

__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    // If the material indicates that the object was a light, "light" the ray
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = -1;
        pathSegment.refractionBefore = false;
    }
    else if (m.hasRefractive > 0.0f) { // totally transparent material
        pathSegment.remainingBounces -= 1;
        if (pathSegment.refractionBefore) {
            pathSegment.refractionBefore = false;
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, 1.0f / m.indexOfRefraction);
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
        }
        else {
            pathSegment.refractionBefore = true;
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, m.indexOfRefraction);
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
        }
    }
    else if (m.hasReflective > 0.0f) { // totally specular reflection
        pathSegment.remainingBounces -= 1;
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + normal * 0.001f;;
        pathSegment.refractionBefore = false;

        if (m.specular.exponent < 0.01f) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            float roughness = m.specular.exponent;

            glm::vec3 wo = -WorldToLocal(normal) * pathSegment.ray.direction;
            glm::vec3 wh = normalize(Sample_wh(wo, rng, roughness));
            glm::vec3 wi = normalize(glm::reflect(-wo, wh));
            glm::vec3 wiW = LocalToWorld(normal) * wi;
            
            if (wo.z * wi.z <= 0.0f) {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
            }
            else {
                
                float cosThetaO = abs(wo.z);
                float cosThetaI = abs(wi.z);
                wh = glm::normalize(wi + wo);

                if (cosThetaO < 1e-5 || cosThetaI < 1e-5 || glm::length(wh) < 1e-5) {
                    pathSegment.color = glm::vec3(0.f);
                    pathSegment.remainingBounces = 0;
                }
                else {
                    float F = 1.0f; // fresnelDielectricEval(dot(wi, wh), 1.0f, m.indexOfRefraction);
                    float D = TrowbridgeReitzD(wh, roughness);
                    float G = TrowbridgeReitzG(wo, wi, roughness);

                    float pdf = D * abs(wh.z) / (4 * dot(wo, wh));
                    if (pdf < 1e-5) {
                        pathSegment.remainingBounces = 0;
                    }
                    else { 
                        pathSegment.color *= m.color * D * F * G / (4.0f * cosThetaO * cosThetaI);
                        pathSegment.color *= abs(glm::dot(wiW, normal)) / pdf;

                        // pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);

                        pathSegment.ray.direction = wiW;
                        // pathSegment.color *= glm::vec3(1.0f, 0.1f, 0.1f);
                    }


                }
            }

            // pathSegment.color *= glm::vec3(1.0f, m.specular.exponent, m.specular.exponent);
        }
    }
    else { // diffuse reflection

        float z;
        glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng, z);

        pathSegment.ray.direction = glm::normalize(wi);
        pathSegment.ray.origin = intersect + normal * 0.001f;
        pathSegment.remainingBounces -= 1;
        pathSegment.color *= m.color * abs(dot(normal, wi)) / z;
        pathSegment.refractionBefore = false;
        // pathSegment.color *= m.color;
    }

}
