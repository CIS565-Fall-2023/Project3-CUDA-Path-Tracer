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

    switch (m.type)
    {
        case MaterialType::DIFFUSE:
		{
            specular = false;
            wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng, pdf));
            return (m.color / PI);
		}
        case MaterialType::SPEC_REFL:
		{
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
            specular = true;
            return sampleSpecularReflectMaterial(m, normal, wo, wi);
		}
        case MaterialType::SPEC_TRANS: 
        {
            specular = true;
            return sampleSpecularTransmissionMaterial(m, normal, wo, wi);
        }
        case MaterialType::SPEC_FRESNEL:
		{
            specular = true;
            return sampleFresnelSpecularMaterial(m, normal, wo, wi, rng);
		}
        case MaterialType::MICROFACET: 
        {
            specular = false;
            return sampleMicrofacetMaterial(m, normal, wo, wi, pdf, rng);
        }
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
      const glm::vec3& point
    , const ShadeableIntersection& intersection
    , const glm::vec3& woW
    , Material* materials
    , Geom* geoms
    , int numGeoms
    , KDAccelNode* nodes
    , int numNodes
    , Light* lights
    , int numLights
    , glm::vec3* envMap
    , int width
    , int height
    , thrust::default_random_engine& rng
) {
    if (numLights == 0) {
        return glm::vec3(0.0f);
    }

    // define variables
    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec3 tangent = intersection.surfaceTangent;
    Material material = materials[intersection.materialId];

    glm::vec3 wiW, Ld(0.0f);
    float lightPdf = 0.0f, scatterPdf = 0.0f;
    bool specular = false;

    // light part
    LightType chosenLightType;
    int chosenLightGeomId;

    glm::vec3 Li = sampleLight(point, normal, wiW, materials, lights, numLights, 
        chosenLightType, lightPdf, chosenLightGeomId, envMap, width, height, rng);

    // test light visibility
    Ray shadowRay;
    shadowRay.origin = point + 0.001f * wiW;
    shadowRay.direction = wiW;
    ShadeableIntersection shadowIntersection;
    shadowIntersection.geomId = -1;

    if (nodes == NULL) {
        computeRayIntersection(geoms, numGeoms, shadowRay, shadowIntersection);
    }
    else {
        computeRayIntersectionFromKdTree(geoms, nodes, numNodes, shadowRay, shadowIntersection);
    }

    if (lightPdf > 0.0f && length(Li) > 0.0f && shadowIntersection.geomId == chosenLightGeomId) {
        // sample bsdf
        glm::vec3 f = getMaterialColor(material, normal, woW, wiW, scatterPdf);
        f *= abs(dot(wiW, normal));
        if (length(f) > 0.0f) {
            Ld += chosenLightType == LightType::AREA ?
                (Li * f * powerHeuristic(1, lightPdf, 1, scatterPdf)) / lightPdf :
                Li * f / lightPdf;
        }
    }

    if (chosenLightType != AREA) {
        return Ld;
    }

    // bsdf part
    glm::vec3 f = sampleMaterial(point, normal, tangent,
        material, woW, wiW, scatterPdf, specular, rng);
    f *= abs(dot(wiW, normal));

    if (length(f) > 0.0 && scatterPdf > 0.0) {
        float weight = specular ? 1.0f : powerHeuristic(1, scatterPdf, 1, lightPdf);

        // test bsdf visibility
        shadowRay.origin = point + 0.001f * wiW;
        shadowRay.direction = wiW;

        shadowIntersection.t = FLT_MAX;
        shadowIntersection.geomId = -1;
        shadowIntersection.materialId = -1;

        if (nodes == NULL) {
            computeRayIntersection(geoms, numGeoms, shadowRay, shadowIntersection);
        }
        else {
            computeRayIntersectionFromKdTree(geoms, nodes, numNodes, shadowRay, shadowIntersection);
        }

        if (shadowIntersection.t > 0) {
            Material lightMat = materials[shadowIntersection.materialId];
            Ld += f * (lightMat.color * lightMat.emittance) * weight / scatterPdf;
        }
    }

    return Ld;
}