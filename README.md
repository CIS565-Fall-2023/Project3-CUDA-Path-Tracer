CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

- [x] Load arbitrary scene(only geom)
    - [x] Triangle
    - [x] ~~Primitive assemble phase~~(This will not work, see `README` of this commit)
    - [x] Use tinygltf
        Remember to check [data type]((https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types)) before using accessor
    ![Alt text](img/accessor_data_types.png)
    - [x] Done with loading a scene with node tree!
            ![blender_reference](img/blender_reference.png)
            ![rendered](img/first_scene.png)
        > Can't tell how excited I am! Now my raytracer is open to most of the scenes!
        - Scene with parenting relationship
            ![with_parenting](img/scene_with_parenting.png)

- [ ] Core
    - [ ] Russian Roulette

- [ ] More BSDF
    - [x] Diffuse
    - [x] Emissive
    - [ ] Reflective
    - [ ] Refractive
    - [ ] Microfacet

- [ ] BVH
    - [x] BoundingBox Array
    - [x] Construct BVH
    - [x] Traverse BVH
    - [ ] Better Heuristics

- [ ] Better sampler
    - [ ] Encapsulate a sampler class
        - Gotta deal with cmake issue
    - [ ] Monte carlo sampling
    - [ ] Importance sampling


- [ ] Light (probably not gonna do a lot about it because gltf has a poor support over area light)

- [ ] Denoiser 
    - [ ] Use Intel OpenImage Denoiser for now


### Log

09.26
Finally, finish gltf loading and basic bsdf.

- A brief trial
    - Note that this difference might be due to different bsdf we are using right now. For convenience, we are using the most naive Diffuse BSDF, while Blender use a standard BSDF by default.
![Alt text](img/first_trial_glb_scene.png)


09.27

Naive BVH (probably done...)
Scene with 1k faces
- One bounce, normal shading
    - Without BVH: FPS 10.9
    ![Alt text](img/normal_shading_without_bvh.png)

    - With BVH: FPS 53.4
    ![Alt text](img/normal_shading_with_bvh.png)
    - 5 times faster
- Multiple bounces
    - Without BVH: FPS 7.6
        ![Alt text](img/multiple_bounces_without_bvh.png)
    - With BVH: FPS 22.8
        ![Alt text](img/multiple_bounces_with_bvh.png)