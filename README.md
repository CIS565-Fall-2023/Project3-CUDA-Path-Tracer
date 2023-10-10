CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Tong Hu
* Tested on: Windows 11, Ryzen 7 1700X @ 3.4GHz 16GB, RTX 2060 6GB (Personal Desktop)

# Visual Improvement
## Different surfaces material

The visualization of light on surfaces is predominantly governed by the material properties of the object. Based on different material properties, Combining with various sampling methods, we can greatly simulate the visual effects produced by the interaction of light with objects. Following two figures from wikipedia described the reflection and refraction of light.

On an ideal diffuse surface, light scatters uniformly in all directions, irrespective of the incident angle. Such surfaces appear soft and non-glossy. Examples might include chalk, unpolished wood, or matte paint. 

A perfectly specular-reflective surface acts like a mirror, reflecting light in a singular direction dictated by the law of reflection. Such surfaces exhibit sharp, clear reflections and are found in materials like polished metals or mirrors. 

A refractive object, like glass or water, can transmit light due to a change in medium and cause refraction. The precise nature of this bending is governed by the material's refractive index. Additionally, such materials can also reflect light.



<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 39%; margin-right: 10%;">
    <img src="./img/reflection_wikipedia.png" style="width: 100%; height: auto;" />
    <p>Reflection of light</p>
  </div>
  
  <div style="flex: 1; max-width: 61%;">
    <img src="./img/refraction_wikipedia.png" style="width: 100%; height: auto;" /> 
    <p>Refraction of light</p>
  </div>
</div>


This can be fixed


<div style="display: flex; justify-content: center; text-align: center;">
  
  <div style="flex: 1; max-width: 33%;">
    <img src="./img/basic_diffusion.png" style="width: 100%; height: auto;" />
    <p>Diffusion</p>
  </div>
  
  <div style="flex: 1; max-width: 33%;">
    <img src="./img/basic_reflection.png" style="width: 100%; height: auto;" /> 
    <p>Reflection</p>
  </div>
  
  <div style="flex: 1; max-width: 33%;">
    <img src="./img/basic_refraction.png" style="width: 100%; height: auto;" /> 
    <p>Refraction</p>
  </div>

</div>

5000 iterations, depth = 15
light emitance: 10


## Anti-aliasing

<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 40%; margin-right: 10%;">
    <img src="./img/without_anti_aliasing_zoomin.png" style="width: 100%; height: auto;" />
    <p>Without Anti-aliasing</p>
  </div>
  
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/anti_aliasing_zoomin.png" style="width: 100%; height: auto;" /> 
    <p>With Anti-aliasing</p>
  </div>
</div>

light emitance:15

## Physically-based depth-of-field

<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/focus_7_6.png" style="width: 100%; height: auto; " />
    <p>focus distance: 7.6</p>
  </div>
  
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/focus_8_7.png" style="width: 100%; height: auto;" /> 
    <p>focus distance: 8.7</p>
  </div>

  <div style="flex: 1; max-width: 40%;">
    <img src="./img/focus_12_6.png" style="width: 100%; height: auto;" /> 
    <p>focus distance: 12.6</p>
  </div>
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/focus_15.png" style="width: 100%; height: auto;" /> 
    <p>focus distance: 115</p>
  </div>
</div>

<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 33%;">
    <img src="./img/aperture_0_5.png" style="width: 100%; height: auto; " />
    <p>focus distance: 10, aperture: 0.5</p>
  </div>
  
  <div style="flex: 1; max-width: 33%;">
    <img src="./img/focus_10.png" style="width: 100%; height: auto;" /> 
    <p>focus distance: 8.7, aperture: 1</p>
  </div>

  <div style="flex: 1; max-width: 33%;">
    <img src="./img/aperture_2.png" style="width: 100%; height: auto;" /> 
    <p>focus distance: 12.6, aperture: 2</p>
  </div>
</div>

## Motion Blur

<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 40%; margin-right: 10%;">
    <img src="./img/static.png" style="width: 100%; height: auto;" />
    <p>Static</p>
  </div>
  
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/motion_blur.png" style="width: 100%; height: auto;" /> 
    <p>Motion Blur</p>
  </div>
</div>

## Subsurface Scattering

<div style="display: flex; justify-content: center; text-align: center;">
  <div style="flex: 1; max-width: 40%; margin-right: 10%;">
    <img src="./img/withou_subsurface_scattering.png" style="width: 100%; height: auto;" />
    <p>Without Subsurface Scattering</p>
  </div>
  
  <div style="flex: 1; max-width: 40%;">
    <img src="./img/subsurface_scattering.png" style="width: 100%; height: auto;" /> 
    <p>Subsurface Scattering</p>
  </div>
</div>

# Performance Improvement
## Stream Compaction


## Material Sorting


## Cache First Bounce


## Core Features
* Ideal diffuse surface, perfectly specular-reflective surface
* Path continuation/termination using Stream Compaction
* Sort by material
* Toggleable option to cache the first bounce



## Other Features:
* Refraction
* Physically-based depth-of-field
* Stochastic Sampled Antialiasing
* Subsurface scattering
* Motion Blur




## Bloopers
(share any images, debug images, etc.)