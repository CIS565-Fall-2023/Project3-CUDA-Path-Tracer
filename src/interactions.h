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

__host__ __device__
glm::vec3 reflectionAndRefraction(glm::vec3 surfaceNormal, glm::vec3 incidentRayDirection, float indexofRefraction, thrust::default_random_engine& rng, bool* underwentRefraction) {
  // Initialize underwentRefraction to false
  *underwentRefraction = false;
  bool is_medium = false;
  glm::vec3 emergentRayDirection;

  // Adjust surface normal and refractive index if light goes from material to air
  if (glm::dot(surfaceNormal, incidentRayDirection) > 0) {
    surfaceNormal = -surfaceNormal;
    is_medium = true;
  }

  // Calculate reflection coefficient using Schlick's approximation
  float R_0 = powf((1.f - indexofRefraction) / (1.f + indexofRefraction), 2.f);
  float reflectionCoefficient = R_0 + (1 - R_0) * powf(1.f + glm::dot(surfaceNormal, incidentRayDirection), 5.f);

  // Generate a random number to decide between reflection and refraction
  thrust::uniform_real_distribution<float> u01(0, 1);
  float randomValue = u01(rng);

  // Reflect or refract based on random value and reflection coefficient
  if (randomValue < reflectionCoefficient) {
    emergentRayDirection = glm::reflect(incidentRayDirection, surfaceNormal);
  }
  else {
    // here, only consider the refraction from material to air(index) or air to material(1/index) for eta.
    if (is_medium) {
			indexofRefraction = 1.f / indexofRefraction;
      emergentRayDirection = glm::refract(incidentRayDirection, surfaceNormal, indexofRefraction);
    }
    else {
			emergentRayDirection = glm::refract(incidentRayDirection, surfaceNormal, 1.f / indexofRefraction);
    }
    

    // Handle total internal reflection
    if (glm::length(emergentRayDirection) == 0.f) {
      emergentRayDirection = glm::reflect(incidentRayDirection, surfaceNormal);
    }
    else {
      *underwentRefraction = true;
    }
  }

  return emergentRayDirection;
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
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    bool is_refraction = false;
    

    if (pathSegment.remainingBounces > 0) {
        glm::vec3 newDir = glm::vec3(0.0f);

        if (m.hasRefractive > 0.0f) {
          pathSegment.color *= m.specular.color;
          newDir = reflectionAndRefraction(normal, pathSegment.ray.direction, m.indexOfRefraction, rng, &is_refraction);
          
        }
        else if (m.hasReflective > 0.0f) {
						newDir = glm::reflect(pathSegment.ray.direction, normal);
						pathSegment.color *= m.specular.color;
				}
				else {
						newDir = calculateRandomDirectionInHemisphere(normal, rng);					
						pathSegment.color *= m.color;
				}

        newDir = glm::normalize(newDir);
        if (is_refraction) {
          pathSegment.ray.origin = intersect + .0002f * pathSegment.ray.direction;
        }
        else {
          pathSegment.ray.origin = intersect;
        }

        pathSegment.ray.direction = newDir;
        pathSegment.remainingBounces--;
      }

}
