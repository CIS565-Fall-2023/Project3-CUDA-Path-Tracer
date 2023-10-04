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

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-1.2, 1.2);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
		);

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
) {
	bool inside = glm::dot(norm, wo) < 0;
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
		float4 color = tex2D<float4>(texs[material.textureId], uv.x, uv.y);
		return glm::vec3(color.x, color.y, color.z);
	}
	if (material.hasRefractive && glm::dot(wi,wo) < 0) {
		return material.color;
	}
	if (material.hasReflective) {
		return material.specular.color;
	}
	return material.color * INV_PI;
}