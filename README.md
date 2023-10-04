# CUDA Path Tracer

<img src="img/cool_renders/029_backrooms_final.png" width="64%"/><img src="img/cool_renders/030_evangelion_final.png" width="28.8%"/>

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Aditya Gupta
  * [Website](http://adityag1.com/), [GitHub](https://github.com/AdityaGupta1), [LinkedIn](https://www.linkedin.com/in/aditya-gupta1/), [3D renders](https://www.instagram.com/sdojhaus/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, NVIDIA GeForce RTX 2070 8GB (personal laptop)
  * Compute capability: 7.5

## Introduction

## Features

### BRDFs

#### Lambertian shading

<img src="img/other/cornell_lambert.png" width="50%">

Basic Lambertian shading, shown here in a Cornell box scene.

#### Specular reflection and refraction

<img src="img/other/cornell_specular.png" width="50%">

Perfect specular reflection and refraction, shown in the same Cornell box scene as above.

Note that this project also has code for microfacet reflection and refraction but it currently does not work properly.

### Scene building tools

#### glTF mesh loading

<img src="img/other/freddy_final.png" width="33%"><img src="img/other/freddy_albedo.png" width="33%"><img src="img/other/freddy_normals.png" width="33%">

These images show, from left to right:
- Final render
- Albedo (base color)
- Normals (with normal map applied)

#### Environment map lights

#### BVH construction and traversal

### Other features

#### Depth of field

<img src="img/other/dof.png" width="50%">

Using a thin lens camera model.

#### Denoising

Using [Intel Open Image Denoise](https://www.openimagedenoise.org/).

## Performance Analysis

### Individual features

#### Sort rays by material type

#### First bounce cache

#### Russian roulette path termination

## Attribution

### Code

- BVH tutorial: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
- Barycentric coordinates: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates/23745#23745
- Normal map: http://www.thetenthplanet.de/archives/1180

### Models

- Freddy Fazbear: https://sketchfab.com/3d-models/fnaf-ar-freddy-fazbear-f6e019333d694cbfbb2f3fbc9e791763
- Chair: https://sketchfab.com/3d-models/metal-folding-chair-c4428a7f4a2f472689a914a3373befc3
- Hair: https://sketchfab.com/3d-models/anime-style-short-hair-c396587aee6a4e63a6476c1400b613eb
- Human character: https://www.mixamo.com/
- Backrooms: https://sketchfab.com/3d-models/simple-backrooms-b7e72135d97b4c81b6c135413e6e2168
- Among Us crewmate: https://sketchfab.com/3d-models/among-us-astronaut-clay-20b591de51eb4fc3a4c5a4d40c6011d5
- Rei plush: https://sketchfab.com/3d-models/rei-ayanami-plushie-0bda564e1d804c2f84950cec57db15d3
