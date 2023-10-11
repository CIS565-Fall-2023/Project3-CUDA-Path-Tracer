CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Utkarsh Dwivedi
  * [LinkedIn](https://www.linkedin.com/in/udwivedi/), [personal website](https://utkarshdwivedi.com/)
* Tested on: Windows 11 Home, AMD Ryzen 7 5800H @ 3.2GHz 16 GB, Nvidia GeForce RTX 3060 Laptop GPU 6 GB

# CUDA Path Tracer

![](img/cornell_highlight1.png)

## Introduction

This is a CUDA implementation of a simple path tracer with bounding volume heirarchy (BVH) based acceleration. The basic path tracer setup shoots a ray from every single pixel in the image resolution (width * height) and traces each ray for `n` bounces, or until the rays hit a light source. At each bounce, luminance information is captured as a throughput value and a final gather on all the rays gets the final colour for each pixel. This is performed using a Monte Carlo estimation of the Light Transport Equation Integral (LTE), which is as follows:


```math
L_O(p,\omega_o) = L_E(p,\omega_o) + \int_{s} f(p,\omega_o,\omega_i)  L_i(p,\omega_i) V(p\prime,p) |dot(\omega_i, N)| \,d\omega_i
```

## Features

### 1. Bidrectional Scattering Distribution Function Computation

**Quick note on terminology**

- BSDF = Bidirectional **Scattering** Distribution Function
- BRDF = Bidirectional **Reflection** Distribution Function
- BTDF = Bidirectional **Transmission** Distribution Function

<ins>**Lambertian Diffuse BRDF, Perfect Specular BRDF, Perfect Refraction BTDF**</ins>

This path tracer supports lambertian diffuse and perfectly specular (reflective), and perfectly transmissive (refractive) materials.

|<img src="img/cornell_diffuse.png" width=400>|<img src="img/cornell_specular.png" width=400>|<img src="img/cornell_refract_only.png" width=400>|
|:-:|:-:|:-:|
|Lambertian Diffuse|Perfect Specular|Perfect Refraction|

In **diffuse BRDF**, light is reflected in all directions, with in the hemisphere with a probability distribution function (PDF) based on cosine weighted hemisphere sampling, i.e., there is a higher probability of light bouncing in a direction that is more aligned with the surface normal. The PDF accounts for any bias that results from the cosine weighting.

In **reflection (specular BRDF)**, light is always reflected in a direction that is the *perfect reflection* along the surface normal.

In **refraction (specular BTDF)**, light is refracted in exactly one direction based on the **indices of refraction** of the two media (one of the object that the light is exiting, and the other of the object that the light is entering). In the case where the angle of incidence is greater than the critical angle, total internal reflection happens, in which case the reflection falls back to **specular BRDF** reflection along the surface normal.

The Index of Refraction (IOR) value of the material can be specified in the `<sceneName>.txt` file under the `MATERIAL` (for example, in the case of glass, this would be `IOR 1.55`). The path tracer assumes that the main transport medium is air, and hard-codes that IOR to 1.

<ins>**Glass and Plastic Materials**</ins>

Additionally, there is support for imperfect specular and refractive materials, using **fresnel computation**. If a material has either  both reflective and refractive (glass) properties, or both reflective and diffuse (plastic) properties, then the ray either performs reflective BSDF calculation, or diffuse/refractive BSDF computation with an equal weight given to both. Specular reflection is only applied to areas on the geometry where the surface normal is not aligned to the view direction (fresnel), and diffuse and refractive BSDFs are applied to the other directions. Since both reflection and diffuse/refraction are only computed half the times, their contributions are boosted twice to account for bias.

|<img src="img/cornell_glass_fresnel.png" width=400>|<img src="img/cornell_plastic.png" width=400>|
|:-:|:-:|
|Glass|Perfect Specular|

**Caustics**

Caustics are a direct result of light refracting through surfaces, as seen above in the glass example image. It becomes more apparent with a more complex shape like an icosphere.

|<img src="img/caustics.png" width=500>|
|:-:|
|Caustics from refraction through glass|

**Fireflies**

Because there is indirect illumination (global illumination) support in this path tracer, some scenes may result in *fireflies*, which are very brightly coloured dots on surfaces where the colours of the dots does not match the material of that surface. This happens because light that bounces off surfaces that have very bright caustics effects will carry illumination from those surfaces. This effect can be seen in this below example, where a sphere is close enough to a box behind it to cause a region of very bright yellow caustics. Any light ray bouncing off of this region carries high luminance with it and deposits that in the form of fireflies.

|<img src="img/fireflies.png" width=500>|
|:-:|
|Fireflies|

### 2. Anti-aliasing with subpixel ray jittering

When rays are generated, instead of shooting them directly through the center of each pixel, the rays are offset by a slight amount. This jittering helps reduce aliasing. The effect can be seen in the below comparison, where the left image has aliased edges on the sphere, but the right one does not.

|<img src="img/cornellWithoutAA.png" width=400>|<img src="img/cornellWithAA.png" width=400>|
|:-:|:-:|
|Without anti-aliasing|With anti-aliasing|

## Performance Analysis

## References

- Adam Mally's CIS 561 Advanced Computer Graphics course at University of Pennsylvania
- TinyGLTF
- PBRT