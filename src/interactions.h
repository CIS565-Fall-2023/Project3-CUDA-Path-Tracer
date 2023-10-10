#pragma once

#include "intersections.h"

#define IMPERFECT_SPECULAR 1

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

    return glm::normalize(up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2);
}

#if IMPERFECT_SPECULAR == 1
// generate perturbed specular sampling direction
// GPU Gems 3 https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
__host__ __device__ glm::vec3 calcPerturbedSpecularDirection(
  glm::vec3 ray_in, // direction of incoming ray
  glm::vec3 normal, // surface normal
  thrust::default_random_engine& rng, // random engine
  float alpha // Phone alpha
) {
  glm::vec3 world_dir(0.0f);
  do {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi1 = u01(rng);
    float xi2 = u01(rng);

    // 
    float theta = acosf(powf(xi1, 1 / (alpha + 1.0f)));
    float phi = 2.0f * PI * xi2;

    // convert to cartesian coordinate where specular direction is the z-axis
    glm::vec3 tangent_dir(cosf(phi) * sinf(theta), sinf(phi)/* * sinf(theta)*/, cosf(theta));
  
    // ideal specular direction
    glm::vec3 specular_dir = glm::reflect(ray_in, normal);

    // Create an orthogonal bases around specular_dir
    glm::vec3 basis_z = specular_dir;
    // find a direction that is not colinear with specular_dir, e.g. world up
    glm::vec3 up = abs(basis_z.z) < 0.9999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 basis_x = glm::normalize(glm::cross(basis_z, up));
    glm::vec3 basis_y = glm::cross(basis_z, basis_x);

    // transform tangent_dir into world space using these bases
    world_dir = glm::normalize(tangent_dir.x * basis_x + tangent_dir.y * basis_y + tangent_dir.z * basis_z);
  } while (glm::dot(world_dir, normal) < 0);
  return world_dir;
}
#endif // IMPERFECT_SPECULAR == 1

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
        bool outside,
        const Material &m,
        thrust::default_random_engine &rng) {
  // TODO: implement this.
  // A basic implementation of pure-diffuse shading will just call the
  // calculateRandomDirectionInHemisphere defined above.

  // not at max depth, decrement
  pathSegment.remainingBounces--;

  // check if this segment hits a light
  if (m.emittance > 0.0f) {
    pathSegment.color *= m.color * m.emittance;
    // ray premature ending at a light source
    pathSegment.remainingBounces = 0;
    return;
  }

  // REFRACTION
  if (m.hasRefractive) {
    // calculate out-going ray directions
    float eta = outside ? 1.0f / m.indexOfRefraction : m.indexOfRefraction / 1.0f;
    glm::vec3 ray_out_reflect = glm::reflect(pathSegment.ray.direction, normal);
    glm::vec3 ray_out_refract = glm::refract(pathSegment.ray.direction, normal, eta);

    // compute coefficient of reflection
    float cosThetaI = glm::dot(pathSegment.ray.direction, -normal);
    float R0 = powf((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction), 2.0f);
    float coefReflection =  R0 + (1.0f - R0) * powf((1.0f - cosThetaI), 5.0f);

    //// choose to sample Specular or Refraction
    thrust::uniform_real_distribution<float> u01(0, 1);
    float choice = u01(rng);
    if (choice < coefReflection) {
      // reflect
      pathSegment.ray.origin = intersect + ray_out_reflect * 0.001f;
      pathSegment.ray.direction = ray_out_reflect;
      pathSegment.color *= m.specular.color;
    }
    else {
      // refract
      if (glm::length(ray_out_refract) < 0.001f) {
        // handle total internal reflection (TIR), use reflection
        pathSegment.ray.direction = ray_out_reflect;
        pathSegment.ray.origin = intersect + ray_out_reflect * 0.001f;
        pathSegment.color *= m.specular.color;
      } else {
        // not TIR
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
        pathSegment.ray.direction = ray_out_refract;
        pathSegment.color *= m.color;
      }
    }
    return;
  }

  // IMPERFECT SPECULAR (Phong Shading)
  if (m.hasReflective) {
#if IMPERFECT_SPECULAR == 1
    // sample both diffusive and specular directions
    glm::vec3 ray_out_diffusive = calculateRandomDirectionInHemisphere(normal, rng);
    float alpha = m.specular.exponent;
    glm::vec3 ray_out_specular = calcPerturbedSpecularDirection(pathSegment.ray.direction, normal, rng, alpha);

    // Calculate BSDF / PDF for both
    float bsdf_diffuse = 1.0f / PI;
    float pdf_diffuse = glm::dot(normal, ray_out_diffusive) / PI;

    glm::vec3 specular_dir = glm::reflect(pathSegment.ray.direction, normal);
    float cosTheta = glm::clamp(glm::abs(glm::dot(specular_dir, ray_out_specular)), 0.0f, 1.0f);
    float bsdf_specular = (alpha + 2) / (2 * PI) * powf(cosTheta, alpha);
    float pdf_specular = (alpha + 1) / (2 * PI) * powf(cosTheta, alpha);

    // calculate MIS weights (power heuristic, beta = 2)
    float mis_weight_sum = powf(pdf_diffuse, 2) + powf(pdf_specular, 2);
    float weight_diffuse = powf(pdf_diffuse, 2) / mis_weight_sum;
    float weight_specular = powf(pdf_specular, 2) / mis_weight_sum;

    // calculate color multipliers
    glm::vec3 color_diffuse = m.color * weight_diffuse * bsdf_diffuse / pdf_diffuse;
    glm::vec3 color_specular = m.specular.color * weight_specular * bsdf_specular / pdf_specular;

    // toss the dice
    thrust::uniform_real_distribution<float> u01(0, 1);
    float choice = u01(rng);

    // impoerfect specular: diffusion + imperfect reflection (Phong)
    if (choice < weight_diffuse) {
      // diffuse branch taken
      pathSegment.ray.direction = ray_out_diffusive;
      pathSegment.ray.origin = intersect + ray_out_diffusive * 0.001f;
      // for lambertian diffusion, BSDF & PDF are aligned (no need to compute)
      pathSegment.color *= color_diffuse;
    }
    else {
      // specular branch taken
      pathSegment.ray.direction = ray_out_specular;
      pathSegment.ray.origin = intersect + ray_out_specular * 0.001f;
      pathSegment.color *= color_specular;
    }
#else
    // original pure specular
    glm::vec3 ray_out = glm::reflect(pathSegment.ray.direction, normal);
    pathSegment.ray.direction = ray_out;
    pathSegment.ray.origin = intersect + ray_out * 0.001f;
    pathSegment.color *= m.specular.color;
#endif
    return;
  }

  // PURE DIFFUSIVE
  // randomize a cosine-weighted ray out
  glm::vec3 ray_out = calculateRandomDirectionInHemisphere(normal, rng);

  // update pathSegment
  pathSegment.ray.direction = ray_out;
  // prevent shadow acne by lifting origin off the surface for a bit
  pathSegment.ray.origin = intersect + ray_out * 0.001f;
  // PDF is aligned with BSDF, no need to calculate PDF/BSDF
  pathSegment.color *= m.color;
}
