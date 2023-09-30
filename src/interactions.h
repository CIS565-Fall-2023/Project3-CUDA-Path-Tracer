#pragma once

#include "intersections.h"

#include "sample.h"
#include "material.h"
#include "light.h"

__host__ __device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (g * g + f * f);
}

__host__ __device__ glm::vec3 sampleMaterial(glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 tangent,
    const Material& m,
    const glm::vec3& wo,
    glm::vec3& wi,
    float& pdf,
    bool& specular,
    thrust::default_random_engine& rng) {

    pdf = 1.0;
    specular = true;
    if (m.hasReflective > 0.0 && m.hasRefractive > 0.0)
    {
        return sampleFresnelSpecularMaterial(m, normal, wo, wi, rng);
    }
    else if (m.hasReflective > 0.0)
    {
        // perfect specular
        return sampleSpecularReflectMaterial(m, normal, wo, wi);

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
        return sampleSpecularTransmissionMaterial(m, normal, wo, wi);
    }
    else
    {
        // diffuse
        specular = false;
        wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng, pdf));
        return (m.color / PI);
    }
}

__host__ __device__ glm::vec3 sampleLight(
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3& wi,
    Light& chosenLight,
    thrust::default_random_engine& rng,
    float& lightPDF,
    Geom* geoms,
    int num_geoms,
    Triangle* triangles,
    int num_triangles,
    Material* materials,
    const Light* lights,
    const int& num_lights) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    int chosenLightIndex = (int)(u01(rng) * num_lights);

    chosenLight = lights[chosenLightIndex];

    switch (chosenLight.lightType)
    {
        case LightType::AREA:
        {
            return sampleAreaLight(chosenLight, intersect, normal, num_lights, wi, lightPDF, 
                geoms, num_geoms, triangles, num_triangles, materials, rng);
        }
    }

    return glm::vec3(0.0f);
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

    float pdf = 1.0;
    glm::vec3 dir = pathSegment.ray.direction;

    pathSegment.throughput *= sampleMaterial(intersect, normal, tangent, m, dir, pathSegment.ray.direction, pdf, pathSegment.isSpecularBounce, rng);
    if (length(pathSegment.throughput) == 0.0f || pdf == 0.0f) {
        pathSegment.remainingBounces = 0;
		return;
    }
    
    pathSegment.throughput *= abs(glm::dot(glm::normalize(pathSegment.ray.direction), normal)) / pdf;

    --pathSegment.remainingBounces;
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
}

__host__ __device__ glm::vec3 sampleUniformLight(
      glm::vec3 point
    , glm::vec3 normal
    , glm::vec3 tangent
    , const glm::vec3& wo
    , Material* materials
    , Material& material
    , Geom* geoms
    , int numGeoms
    , Triangle* triangles
    , int numTriangles
    , Light* lights
    , int numLights
    , thrust::default_random_engine& rng
) {
    glm::vec3 Ld(0.0f);
    // sample light source
    glm::vec3 wiW;
    float lightPdf = 0.0f, scatterPdf = 0.0f;
    bool specular = false;
    Light chosenLight;
    glm::vec3 lightColor = sampleLight(point, normal,
        wiW, chosenLight, rng, lightPdf, geoms, numGeoms, triangles, numTriangles, materials, lights, numLights);

    glm::vec3 f = sampleMaterial(point, normal, tangent, material, wo, wiW, scatterPdf, specular, rng);
    f *= abs(dot(wiW, normal));

    if (length(lightColor) > 0.0f && !specular) {
        if (lightPdf > 0.0f && length(f) > 0.0f) {
            Ld += chosenLight.lightType == LightType::AREA ?
                (f * lightColor * powerHeuristic(1, lightPdf, 1, scatterPdf)) / lightPdf :
                f * lightColor / lightPdf;
        }
    }

    // sample BSDF
    if (chosenLight.lightType == LightType::AREA) {
        float weight = specular ? 1.0f : powerHeuristic(1, scatterPdf, 1, lightPdf);

        if (scatterPdf > 0.0f && length(f) > 0.0f) {
            // check visibility
            Ray shadowRay;
            shadowRay.origin = point;
            shadowRay.direction = wiW;
            ShadeableIntersection shadowIntersection;
            computeRayIntersection(geoms, numGeoms, triangles, numTriangles, shadowRay, shadowIntersection);

            if (shadowIntersection.t > 0.0f && shadowIntersection.materialId == chosenLight.geom.materialid) {
                Ld += f * lightColor * weight / scatterPdf;
            }
        }
    }

    return Ld;
}