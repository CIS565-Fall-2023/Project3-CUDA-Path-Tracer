# CUDA Path Tracer

<img src="img/cool_renders/029_backrooms_final.png" width="64%"/><img src="img/cool_renders/030_evangelion_final.png" width="28.8%"/>

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Aditya Gupta
  * [Website](http://adityag1.com/), [GitHub](https://github.com/AdityaGupta1), [LinkedIn](https://www.linkedin.com/in/aditya-gupta1/), [3D renders](https://www.instagram.com/sdojhaus/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, NVIDIA GeForce RTX 2070 8GB (personal laptop)
  * Compute capability: 7.5

## Introduction

TODO

## Features

### BRDFs

#### Lambertian shading

<img src="img/other/cornell_lambert.png" width="50%">

Basic Lambertian shading, shown here in a Cornell box scene.

#### Specular reflection and refraction

<img src="img/other/cornell_specular.png" width="50%">

Perfect specular reflection and refraction, shown in the same Cornell box scene as above.

Note that this project also has code for microfacet reflection and refraction but it currently does not work properly. The render on the right at the top of the page does use a roughness value of 0.3 (for the transparent character) but the calculations aren't entirely accurate even if the effect looks cool.

### Scene building tools

#### glTF mesh loading

<img src="img/other/freddy_final.png" width="33%"><img src="img/other/freddy_albedo.png" width="33%"><img src="img/other/freddy_normals.png" width="33%">

Using [tinygltf](https://github.com/syoyo/tinygltf). The above images show, from left to right:
- Final render
- Albedo (base color)
- Normals (with normal map applied)

#### Environment map lights

<img src="img/other/freddy_env_map.png" width="50%">

#### BVH construction and traversal

The [above scene](#environment-map-lights) renders at 26.4 FPS using a BVH and 2.9 FPS without a BVH. The mesh has 17,386 triangles and it takes about 11.5 ms to construct the BVH.

More detailed performance analysis is given [below](#bvh).

### Other features

#### Depth of field

<img src="img/other/dof.png" width="50%">

Using a thin lens camera model.

#### Denoising

<img src="img/other/backrooms_20_spp.png" width="50%"><img src="img/other/backrooms_20_spp_denoised.png" width="50%"/>

Using [Intel Open Image Denoise](https://www.openimagedenoise.org/). Left is raw output and right is denoised. Both images were taken after 20 samples per pixel (~8 seconds). Denoising also takes albedo and normals into account (see example auxiliary images [above](#gltf-mesh-loading)).

## Performance Analysis

TODO

### Partition rays

TODO

- Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.
- Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

### Sort rays by material type

TODO

### First bounce cache

TODO

- Provide performance benefit analysis across different max ray depths.

### Russian roulette path termination

TODO

### BVH

TODO

## Bloopers

<img src="img/cool_renders/002_sorting_intersections.png" width="50%">

This appeared when I sorted intersections by material ID without also sorting the path segments in tandem.

<img src="img/cool_renders/005_idk.png" width="50%">

Really not sure where this came from, but it appeared while I was working on the first bounce cache.

<img src="img/cool_renders/011_monkey_attempt_1.png" width="50%">

Evil monkey.

<img src="img/cool_renders/016_idk.png" width="50%">

Seems like the monkey exploded?

<img src="img/cool_renders/019_monkey_broke.png" width="50%">

The monkey is empty inside, just like me after spending two weeks on this assignment.

<img src="img/cool_renders/023_freddy_artifacts.png" width="50%">

🤨 (I was trying to fix the shading artifacts on the eye sockets.)

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
