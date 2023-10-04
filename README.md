# CUDA Path Tracer

![](img/cool_renders/029_backrooms_final.png)

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Aditya Gupta
  * [Website](http://adityag1.com/), [GitHub](https://github.com/AdityaGupta1), [LinkedIn](https://www.linkedin.com/in/aditya-gupta1/), [3D renders](https://www.instagram.com/sdojhaus/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, NVIDIA GeForce RTX 2070 8GB (personal laptop)
  * Compute capability: 7.5

### Features

- BRDFs
  - Lambertian shading
  - Specular reflection
  - Specular refraction
- Performance options
  - Toggleable option to sort rays by material type
  - Toggleable first bounce cache
  - Russian roulette path termination
- Scene building tools
  - glTF mesh loading
    - Diffuse and emission texture loading
  - Environment map lights
  - BVH construction and traversal
- Other features
  - Depth of field
  - Denoising using [Intel Open Image Denoise](https://www.openimagedenoise.org/)

Note that there is also code for microfacet reflection and refraction but it currently does not work properly.

### Attribution

#### Code

- BVH tutorial: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
- Barycentric coordinates: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates/23745#23745
- Normal map: http://www.thetenthplanet.de/archives/1180

#### Models

- Freddy Fazbear: https://sketchfab.com/3d-models/fnaf-ar-freddy-fazbear-f6e019333d694cbfbb2f3fbc9e791763
- Chair: https://sketchfab.com/3d-models/metal-folding-chair-c4428a7f4a2f472689a914a3373befc3
- Hair: https://sketchfab.com/3d-models/anime-style-short-hair-c396587aee6a4e63a6476c1400b613eb
- Human character: https://www.mixamo.com/
- Backrooms: https://sketchfab.com/3d-models/simple-backrooms-b7e72135d97b4c81b6c135413e6e2168
- Among Us crewmate: https://sketchfab.com/3d-models/among-us-astronaut-clay-20b591de51eb4fc3a4c5a4d40c6011d5
- Rei plush: https://sketchfab.com/3d-models/rei-ayanami-plushie-0bda564e1d804c2f84950cec57db15d3