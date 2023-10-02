#pragma once

#include "intersections.h"
#include "utilities.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    const float up = sqrt(u01(rng)); // cos(theta)
    const float over = sqrt(1 - up * up); // sin(theta)
    const float around = u01(rng) * TWO_PI;

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
    const glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    const glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
void applyReflection(PathSegment& pathSegment, Ray& newRay, const glm::vec3& normal, const Material& m)
{
    newRay.direction = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.color *= m.specular.color;
}

__host__ __device__
void applyRefraction(PathSegment& pathSegment, Ray& newRay, const glm::vec3& normal, const Material& m)
{
    const bool entering = glm::dot(pathSegment.ray.direction, normal) < 0;
    newRay.direction = glm::refract(pathSegment.ray.direction, entering ? normal : -normal, 
        entering ? 1.0f / m.specular.indexOfRefraction : m.specular.indexOfRefraction);
    pathSegment.color *= m.specular.color;
}

__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
    const glm::vec2 uOffset = 2.f * u - glm::vec2(1);

    if (uOffset.x == 0 && uOffset.y == 0)
    {
        return glm::vec2(0);
    }

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
    } 
    else
    {
        r = uOffset.y;
        theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(cos(theta), sin(theta));
}

__device__ glm::vec3 tex2DCustom(cudaTextureObject_t tex, glm::vec2 uv)
{
    //uchar4 texCol = tex2D<uchar4>(tex, uv.x, uv.y);
    //return glm::vec3(texCol.x, texCol.y, texCol.z) / 255.f;
    const float4 texCol = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(texCol.x, texCol.y, texCol.z);
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
    PathSegment& pathSegment,
    ShadeableIntersection& isect,
    const glm::vec3& isectPos,
    const Geom* const geoms,
    const Triangle* const tris,
    const Material &m,
    const cudaTextureObject_t* const textureObjects,
    thrust::default_random_engine &rng) 
{
    glm::vec3 diffuseColor;
    if (m.diffuse.textureIdx != -1)
    {
        diffuseColor = tex2DCustom(textureObjects[m.diffuse.textureIdx], isect.uv);
    }
    else
    {
        diffuseColor = m.diffuse.color;
    }

    glm::vec3 normal;
    if (m.normalMap.textureIdx != -1)
    {
        // http://www.thetenthplanet.de/archives/1180
        const Triangle& tri = tris[isect.triIdx];
        const Vertex& v0 = tri.v0;
        const Vertex& v1 = tri.v1;
        const Vertex& v2 = tri.v2;

        const glm::mat4 transform = geoms[isect.hitGeomIdx].transform;

        const glm::vec3 v0Pos = multiplyMV(transform, glm::vec4(v0.pos, 1));
        const glm::vec3 v1Pos = multiplyMV(transform, glm::vec4(v1.pos, 1));
        const glm::vec3 v2Pos = multiplyMV(transform, glm::vec4(v2.pos, 1));

        const glm::vec3 dp1 = v1Pos - v0Pos;
        const glm::vec3 dp2 = v2Pos - v0Pos;
        const glm::vec2 duv1 = v1.uv - v0.uv;
        const glm::vec2 duv2 = v2.uv - v0.uv;

        const glm::vec3& N = isect.surfaceNormal;

        const glm::vec3 dp2perp = glm::cross(dp2, N);
        const glm::vec3 dp1perp = glm::cross(N, dp1);
        const glm::vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        const glm::vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

        const float invmax = 1.f / sqrt(max(glm::dot(T, T), glm::dot(B, B)));
        const glm::mat3 TBN = glm::mat3(T * invmax, B * invmax, N);

        const glm::vec3 normalMapCol = tex2DCustom(textureObjects[m.normalMap.textureIdx], isect.uv);
        const glm::vec3 mappedNormal = glm::normalize(2.0f * normalMapCol - 1.0f);
        normal = glm::normalize(TBN * mappedNormal);
    }
    else
    {
        normal = isect.surfaceNormal;
    }
    isect.surfaceNormal = normal;

    const float diffuseLuminance = Utils::luminance(diffuseColor);
    const float specularLuminance = Utils::luminance(m.specular.color);
    const float diffuseChance = diffuseLuminance / (diffuseLuminance + specularLuminance); // XXX: bad if both luminances are 0 (pure black material)

    thrust::uniform_real_distribution<float> u01(0, 1);

    Ray newRay;
    if (u01(rng) < diffuseChance)
    {
        // diffuse
        newRay.direction = calculateRandomDirectionInHemisphere(normal, rng); // XXX: make normal face same direction as ray? (e.g. for inside of sphere)
        pathSegment.color *= diffuseColor / (diffuseChance);

        /* 
        disabled because lambert term is canceled out by the PDF
        https://computergraphics.stackexchange.com/questions/9499/confusion-around-lamberts-cosine-law-in-ray-tracing-in-one-weekend
        */ 
        // pathSegment.diffuse.color *= glm::dot(newRay.direction, normal);
    } 
    else
    {
        if (m.specular.hasReflective > 0 && m.specular.hasRefractive > 0)
        {
            const float cosTheta = abs(glm::dot(pathSegment.ray.direction, normal));
            float R0 = (1 - m.specular.indexOfRefraction) / (1 + m.specular.indexOfRefraction);
            R0 = R0 * R0;
            const float fresnel = R0 + (1 - R0) * pow(1.f - cosTheta, 5.f);

            if (u01(rng) < fresnel) // implicit multiplication by fresnel
            {
                applyReflection(pathSegment, newRay, normal, m);
            } 
            else
            {
                applyRefraction(pathSegment, newRay, normal, m);
            }
        } 
        else if (m.specular.hasReflective > 0)
        {
            applyReflection(pathSegment, newRay, normal, m);
        } 
        else if (m.specular.hasRefractive > 0)
        {
            applyRefraction(pathSegment, newRay, normal, m);
        }

        if (glm::dot(newRay.direction, newRay.direction) == 0)
        {
            pathSegment.color = glm::vec3(0);
            pathSegment.remainingBounces = 0;
            return;
        }
        else
        {
            pathSegment.color /= (1 - diffuseChance);
        }
    }

    newRay.origin = isectPos + 0.0001f * newRay.direction;
    pathSegment.ray = newRay;
}
