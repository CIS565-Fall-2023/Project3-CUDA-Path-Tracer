#pragma once

#include "intersections.h"
#include "utilities.h"

__host__ __device__ glm::vec3 calculateDirectionNotNormal(const glm::vec3 normal)
{
    // Find alpha direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        return glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        return glm::vec3(0, 1, 0);
    }
    else
    {
        return glm::vec3(0, 0, 1);
    }
}

/**
 * Computes alpha cosine-weighted random direction in alpha hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    const float up = sqrt(u01(rng)); // cos(theta)
    const float over = sqrt(1 - up * up); // sin(theta)
    const float around = u01(rng) * TWO_PI;

    // Use not-normal direction to generate two perpendicular directions
    const glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, calculateDirectionNotNormal(normal)));
    const glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ float cosSquaredTheta(const glm::vec3 w)
{
    return w.z * w.z;
}

__device__ float cosTheta(const glm::vec3 w)
{
    return w.z;
}

__device__ float absCosTheta(const glm::vec3 w)
{
    return fabs(w.z);
}

__device__ float sinSquaredTheta(const glm::vec3 w)
{
    return fmax(0.f, 1.f - cosSquaredTheta(w));
}

__device__ float sinTheta(const glm::vec3 w)
{
    return sqrtf(sinSquaredTheta(w));
}

__device__ float tanSquaredTheta(const glm::vec3 w)
{
    return sinSquaredTheta(w) / cosSquaredTheta(w);
}

__device__ float tanTheta(const glm::vec3 w)
{
    return sinTheta(w) / cosTheta(w);
}

__device__ float D_GGX(const glm::vec3 wh, const float alpha)
{
    float tan2Theta = tanSquaredTheta(wh);
    if (isinf(tan2Theta)) return 0.f;
    const float cos4Theta = cosSquaredTheta(wh) * cosSquaredTheta(wh);
    float e = tan2Theta / (alpha * alpha);
    return 1.f / (PI * alpha * alpha * cos4Theta * (1.f + e) * (1.f + e));
}

__device__ float lambda_GGX(const glm::vec3 w, const float alpha)
{
    float absTanTheta = fabs(tanTheta(w));
    if (isinf(absTanTheta)) return 0.f;
    float squaredAlphaTanTheta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1.f + sqrtf(1.f + squaredAlphaTanTheta)) * 0.5f;
}

__device__ float G_GGX(const glm::vec3 wo, const glm::vec3 wi, const float alpha)
{
    return 1.f / (1.f + lambda_GGX(wo, alpha) + lambda_GGX(wi, alpha));
}

__device__ float schlickFresnel(const glm::vec3& V, const glm::vec3& N, const float ior)
{
    float cosTheta = abs(glm::dot(V, N));
    float R0 = (1 - ior) / (1 + ior);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1.f - cosTheta, 5.f);
}

__device__ glm::vec3 sampleH_GGX(const glm::vec3 wo, const float alpha, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi1 = u01(rng);
    float xi2 = u01(rng);

    glm::vec3 wh;
    float cosTheta = 0, phi = TWO_PI * xi1;
    float tanTheta2 = alpha * alpha * xi2 / (1.f - xi2);
    cosTheta = rsqrtf(1.f + tanTheta2);
    float sinTheta = sqrtf(fmax(0.f, 1.f - cosTheta * cosTheta));

    wh = glm::vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
    if (glm::dot(wo, wh) < 0) wh = -wh;
    return wh;
}

__device__
void applyReflection(PathSegment& pathSegment, Ray& newRay, const glm::vec3 N, const Material& m, thrust::default_random_engine& rng,
    bool isSingular)
{
    pathSegment.color *= m.specular.color;

    if (m.specular.roughness == 0)
    {
        newRay.direction = glm::reflect(pathSegment.ray.direction, N);

        if (!isSingular)
        {
            float fresnel = schlickFresnel(pathSegment.ray.direction, N, m.specular.indexOfRefraction);
            pathSegment.color *= fresnel;
        }

        return;
    }

    const float alpha = m.specular.roughness * m.specular.roughness;

    const glm::vec3 T = glm::normalize(glm::cross(N, calculateDirectionNotNormal(N)));
    const glm::vec3 B = glm::normalize(glm::cross(N, T));

    const glm::vec3 V = glm::normalize(-glm::vec3(glm::dot(pathSegment.ray.direction, T), glm::dot(pathSegment.ray.direction, B), glm::dot(pathSegment.ray.direction, N)));
    const glm::vec3 H = glm::normalize(sampleH_GGX(V, alpha, rng));
    const glm::vec3 L = glm::normalize(glm::reflect(-V, H));
    
    /*
    float cosThetaO = absCosTheta(V);
    float cosThetaI = absCosTheta(L);

    //if (glm::dot(V, L) < 0 || cosThetaI == 0 || cosThetaO == 0 || (H.x == 0 && H.y == 0 && H.z == 0))
    //{
    //    pathSegment.color = glm::vec3(0);
    //    pathSegment.remainingBounces = 0;
    //    return;
    //}

    const float G = G_GGX(V, L, alpha);
    const float D = D_GGX(H, alpha);

    const float R = G * D / (4.f * cosThetaI * cosThetaO);
    const float pdf = D * absCosTheta(H) / (4.f * glm::dot(V, H));
    pathSegment.color *= R / pdf;
    */

    newRay.direction = L.x * T + L.y * B + L.z * N;

    if (!isSingular)
    {
        float fresnel = schlickFresnel(V, H, m.specular.indexOfRefraction);
        pathSegment.color *= fresnel;
    }
}

__device__
void applyRefraction(PathSegment& pathSegment, Ray& newRay, glm::vec3 N, const Material& m, thrust::default_random_engine& rng,
    bool isSingular)
{
    pathSegment.color *= m.specular.color;

    if (m.specular.roughness == 0)
    {
        bool entering = glm::dot(pathSegment.ray.direction, N) < 0;
        if (entering)
        {
            newRay.direction = glm::refract(pathSegment.ray.direction, N, 1.f / m.specular.indexOfRefraction);
        }
        else
        {
            newRay.direction = glm::refract(pathSegment.ray.direction, -N, m.specular.indexOfRefraction);
        }

        if (!isSingular)
        {
            float fresnel = schlickFresnel(pathSegment.ray.direction, N, m.specular.indexOfRefraction);
            pathSegment.color *= (1.f - fresnel);
        }

        return;
    }

    const float alpha = m.specular.roughness * m.specular.roughness;

    bool entering = glm::dot(pathSegment.ray.direction, N) < 0;
    if (entering)
    {
        N = -N;
    }

    const glm::vec3 T = glm::normalize(glm::cross(N, calculateDirectionNotNormal(N)));
    const glm::vec3 B = glm::normalize(glm::cross(N, T));

    const glm::vec3 V = glm::normalize(-glm::vec3(glm::dot(pathSegment.ray.direction, T), glm::dot(pathSegment.ray.direction, B), glm::dot(pathSegment.ray.direction, N)));
    glm::vec3 H = glm::normalize(sampleH_GGX(V, alpha, rng));
    glm::vec3 L = glm::normalize(glm::refract(-V, H, entering ? 1.f / m.specular.indexOfRefraction : m.specular.indexOfRefraction));

    /*
    float cosThetaO = absCosTheta(V);
    float cosThetaI = absCosTheta(L);

    //if (glm::dot(V, L) < 0 || cosThetaI == 0 || cosThetaO == 0 || (H.x == 0 && H.y == 0 && H.z == 0))
    //{
    //    pathSegment.color = glm::vec3(0);
    //    pathSegment.remainingBounces = 0;
    //    return;
    //}

    const float G = G_GGX(V, L, alpha);
    const float D = D_GGX(H, alpha);

    const float R = G * D / (4.f * cosThetaI * cosThetaO);
    const float pdf = D * absCosTheta(H) / (4.f * glm::dot(V, H));
    pathSegment.color *= R / pdf;
    */

    newRay.direction = L.x * T + L.y * B + L.z * N;

    if (!isSingular)
    {
        float fresnel = schlickFresnel(V, H, m.specular.indexOfRefraction);
        pathSegment.color *= (1.f - fresnel);
    }
}

__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
    glm::vec2 uOffset = 2.f * u - glm::vec2(1);

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
    float4 texCol = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(texCol.x, texCol.y, texCol.z);
}

/**
 * Scatter alpha ray with some probabilities according to the material properties.
 * For example, alpha diffuse surface scatters in alpha cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in alpha few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between alpha each effect (alpha diffuse bounce
 *   and alpha specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as alpha good starting point - it
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
    const Material& m,
    const cudaTextureObject_t* const textureObjects,
    thrust::default_random_engine& rng)
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

    if (m.normalMap.textureIdx != -1)
    {
        // http://www.thetenthplanet.de/archives/1180
        const Triangle& tri = tris[isect.triIdx];
        const Vertex& v0 = tri.v0;
        const Vertex& v1 = tri.v1;
        const Vertex& v2 = tri.v2;

        glm::mat4 transform = geoms[isect.hitGeomIdx].transform;

        glm::vec3 v0Pos = multiplyMV(transform, glm::vec4(v0.pos, 1));
        glm::vec3 v1Pos = multiplyMV(transform, glm::vec4(v1.pos, 1));
        glm::vec3 v2Pos = multiplyMV(transform, glm::vec4(v2.pos, 1));

        glm::vec3 dp1 = v1Pos - v0Pos;
        glm::vec3 dp2 = v2Pos - v0Pos;
        glm::vec2 duv1 = v1.uv - v0.uv;
        glm::vec2 duv2 = v2.uv - v0.uv;

        glm::vec3 N = isect.surfaceNormal;

        glm::vec3 dp2perp = glm::cross(dp2, N);
        glm::vec3 dp1perp = glm::cross(N, dp1);
        glm::vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        glm::vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

        float invmax = rsqrtf(max(glm::dot(T, T), glm::dot(B, B)));
        glm::mat3 TBN = glm::mat3(T * invmax, B * invmax, N);

        glm::vec3 normalMapCol = tex2DCustom(textureObjects[m.normalMap.textureIdx], isect.uv);
        glm::vec3 mappedNormal = glm::normalize(2.0f * normalMapCol - 1.0f);
        isect.surfaceNormal = glm::normalize(TBN * mappedNormal);
    }

    float diffuseLuminance = Utils::luminance(diffuseColor);
    float specularLuminance = Utils::luminance(m.specular.color);
    float diffuseChance = diffuseLuminance / (diffuseLuminance + specularLuminance); // XXX: bad if both luminances are 0 (pure black material)

    thrust::uniform_real_distribution<float> u01(0, 1);

    Ray newRay;
    if (u01(rng) < diffuseChance)
    {
        // diffuse
        newRay.direction = calculateRandomDirectionInHemisphere(isect.surfaceNormal, rng); // XXX: make normal face same direction as ray? (e.g. for inside of sphere)
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
            if (u01(rng) < 0.5f)
            {
                applyReflection(pathSegment, newRay, isect.surfaceNormal, m, rng, false);
            }
            else
            {
                applyRefraction(pathSegment, newRay, isect.surfaceNormal, m, rng, false);
            }

            pathSegment.color *= 2.f;
        }
        else if (m.specular.hasReflective > 0)
        {
            applyReflection(pathSegment, newRay, isect.surfaceNormal, m, rng, true);
        }
        else if (m.specular.hasRefractive > 0)
        {
            applyRefraction(pathSegment, newRay, isect.surfaceNormal, m, rng, true);
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

    newRay.origin = isectPos + EPSILON * newRay.direction;
    pathSegment.ray = newRay;
}
