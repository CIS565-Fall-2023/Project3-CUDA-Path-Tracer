#pragma once

#include "intersections.h"
#include "cuda_runtime.h"

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
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}


__device__ glm::vec2 sampleUniformDisk(thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u11(-1.f, 1.f);
	glm::vec2 minus1To1(u11(rng), u11(rng));
	if (minus1To1.x == 0 && minus1To1.y == 0) {
		return glm::vec2(0.f);
	}
	float theta, r;
	if (abs(minus1To1.x) > abs(minus1To1.y)) {
		r = minus1To1.x;
		theta = PI * 0.25 * (minus1To1.y / minus1To1.x);
	}
	else {
		r = minus1To1.y;
		theta = PI * 0.5 - PI * 0.25 * (minus1To1.x / minus1To1.y);
	}
	return r * glm::vec2(cos(theta), sin(theta));
}
/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
	Camera cam
	, int iter
	, int traceDepth
	, float lenRadius
	, float focusLen
	, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		glm::mat3 camToWorld(cam.right, cam.up, cam.view);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
		thrust::uniform_real_distribution<float> u01(-1.2, 1.2);

		glm::vec2 clip = 2.f * (glm::vec2(((float)x + u01(rng))/ float(cam.resolution.x), ((float)y + u01(rng)) / float(cam.resolution.y)) - glm::vec2(0.5));
		float tanX = tan(cam.fov.x * (PI / 360));
		float tanY = tan(cam.fov.y * (PI / 360));

		glm::vec3 dir = glm::normalize(glm::vec3(-clip.x * tanX, -clip.y * tanY, 1.f));
		if (lenRadius > 0) {
			glm::vec2 lensPt = sampleUniformDisk(rng) * lenRadius;
			float ft = focusLen / dir.z;
			glm::vec3 focusPt = dir * ft;
			focusPt = camToWorld * (focusPt - glm::vec3(lensPt,0));
			segment.ray.origin = cam.position + camToWorld * glm::vec3(lensPt,0);
			segment.ray.direction = glm::normalize(focusPt);
		}
		else {
			segment.ray.origin = cam.position;
			segment.ray.direction = camToWorld * dir;
		};
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__device__ glm::vec3 sampleHemisphereCosine(glm::vec2 i, float& pdf) {
	float sinTheta = sqrt(1.0 - i.x);
	float cosTheta = sqrt(i.x);
	float sinPhi = sin(TWO_PI * i.y);
	float cosPhi = cos(TWO_PI * i.y);
	float x = sinTheta * cosPhi;
	float y = sinTheta * sinPhi;
	float z = cosTheta;
	pdf = z * INV_PI;
	return glm::vec3(x, y, z);
}

__device__ glm::mat3 tangentToWorld(glm::vec3 worldNorm) {
	glm::vec3 up = abs(worldNorm.z) < 0.999 ? glm::vec3(0.0, 0.0, 1.0) : glm::vec3(1.0, 0.0, 0.0);
	glm::vec3 tangent = normalize(cross(up, worldNorm));
	glm::vec3 bitangent = cross(worldNorm, tangent);
	return glm::mat3(tangent, bitangent, worldNorm);
}

__device__ bool sampleRay(
	glm::vec3 wo
	, glm::vec3 norm
	, Material material
	, thrust::default_random_engine& rng
	, float& out_pdf
	, glm::vec3& out_wi
	, PathSegment& seg
) {
	bool inside = glm::dot(norm, wo) <0;
	glm::vec3 normal = norm;
	if (inside) normal = -normal;
	if (material.hasRefractive) {
		float indexOfRefract = 1.f/material.indexOfRefraction;
		float R0 = pow((1 - indexOfRefract) / (1 + indexOfRefract), 2.);

		thrust::uniform_real_distribution<float> u01(0.f, 1.f);
		
		if (inside) {
			indexOfRefract = 1.0 / indexOfRefract;
		}
		float fresnel = R0 + (1 - R0) * pow(1 - glm::dot(normal, wo), 5);//how much is reflected
		glm::vec3 refract = glm::refract(-wo, normal, indexOfRefract);
		if (
			!isnan(refract.x) &&
			fresnel < u01(rng)
			) // not reflected
		{ 
			if (inside)seg.inScatterMedium = false;
			float absdot = glm::dot(-normal, refract);
			out_pdf = absdot;
			out_wi = refract;
			return true;
		}
	}
	if (material.hasReflective) { 
		glm::vec3 reflect = glm::reflect(-wo, normal);
		float absdot = glm::dot(normal, reflect);
		out_pdf = absdot;
		out_wi = reflect;
		return true;
	}
	else {
		
		thrust::uniform_real_distribution<float> u01(0.f, 1.f);
		glm::vec3 sampleDir = sampleHemisphereCosine(glm::vec2(u01(rng), u01(rng)), out_pdf);
		glm::mat3 rotMat = tangentToWorld(normal);
		out_wi = rotMat * sampleDir;
		return true;
	}
	return false;
}

__device__ glm::vec3 texture2D(
	glm::vec2 uv
	, cudaTextureObject_t tex
) {
	float4 color = tex2D<float4>(tex, uv.x, uv.y);
	return glm::vec3(color.x, color.y, color.z);
}

__host__ __device__
void scatterRay(
	PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
}

__device__ glm::vec3 getBSDF(
	glm::vec3 wi
	, glm::vec3 wo
	, glm::vec2 uv
	, Material material
	, cudaTextureObject_t* texs
) {
	if (material.textureId != -1) {
		return glm::pow(texture2D(uv, texs[material.textureId]),glm::vec3(2.2)) * INV_PI;
	}
	if (material.hasReflective) {
		return material.specular.color;
	}
	return material.color * INV_PI;
}

__device__ glm::vec3 getEnvLight(
	glm::vec3 wi
	, cudaTextureObject_t envMap
) {
	glm::vec2 uv = glm::vec2(atan2f(wi.z, wi.x), asin(wi.y)) * glm::vec2(0.1591, 0.3183) + 0.5f;
	return texture2D(uv, envMap);
}

//return hit surface
__device__ bool marchRay(
	PathSegment& segment
	, ShadeableIntersection& intersect
	, Material material
	, float scatteringDistance
	, float absorbAtDistance
	, thrust::default_random_engine& rng
) {
	segment.inScatterMedium = segment.inScatterMedium || (material.isScatterMedium && glm::dot(intersect.surfaceNormal, segment.ray.direction) > 0);
	if (segment.inScatterMedium) {
		thrust::uniform_real_distribution<float> u01(0.f, 1.f);
		float distFactor = -logf(u01(rng));
		float dist = distFactor * scatteringDistance;
		glm::vec3 absorbtCoefficient = -log(glm::vec3(0,.98,.98)) / absorbAtDistance;
		if (dist <  intersect.t) {
			float pdf = exp(-distFactor);
			segment.color /= pdf;
			segment.color *= exp(-absorbtCoefficient * dist);//absorb during light transmission
			segment.ray.origin += (segment.ray.direction * dist);
			return false;
		}
		else {
			segment.color *= exp(-absorbtCoefficient * intersect.t);//absorb during light transmission
			segment.ray.origin += (segment.ray.direction * intersect.t);
			return true;
		}
	}
	else {
		segment.ray.origin += (segment.ray.direction * intersect.t);
	}
	return true;
}