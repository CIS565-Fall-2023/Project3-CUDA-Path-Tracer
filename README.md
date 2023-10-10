CUDA Path Tracer
================
[pbrt]: https://pbrt.org/

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yian Chen
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)
<div align="center">
    <img src="img/emissive_robot_car_2.png" width="45%"/>
    <img src="img/result_bunny.png" width="45%"/>
    <img src="img/bump_mapping_after.png" width="45%"/>
    <img src="img/robots.png" width="45%"/>
    <img src="img/indoor.png" width="45%"/>
</div>

### Implemeted Feature 

- [x] Core(as required in Project 3)
    - [x] Diffuse & Specular
    - [x] Jittering (Antialiasing)
    - [x] First Bounce Cache
    - [x] Sort by material
- [x] Load gltf
- [x] BVH
- [x] Texture mapping & bump mapping
- [x] Environment Mapping
- [x] Microfacet BSDF
- [x] Emissive BSDF (with Emissive Texture)
- [x] Direct Lighting
- [x] Multiple Importance Sampling
- [x] Depth of Field
- [x] Tone mapping && Gamma Correction


### Core features (As required by project instruction)
- Diffuse & Specular

![Diffuse & Specular Demo]()

- Jittering
<table>
    <tr>
        <th>Before jittering</th>
        <th>After jittering</th>
    </tr>
    <tr>
        <th><img src="./img/sampler_indep.jpg"/></th>
        <th><img src="./img/sampler_sobol.jpg"/></th>
    </tr>
</table>

- 


### BVH
On host, we can construct and traverse BVH recursively. While in this project, our code run on GPU. While recent cuda update allows recursive function execution on device, we cannot take that risk as raytracer is very performance-oriented. Recursive execution will slow down the kernel function, as it may bring dynamic stack size. 

Thanks to , a novel BVH constructing and traversing algorithm is adopted in this pathtracer. 

This pathtracer only implements a simple version of MTBVH. Instead of constructing 6 BVHs and traversing one of them at runtime, only 1 BVH is constructed. *It implies that this pathtracer still has the potential of speeding up*.

- With BVH & Without BVH:


### Texture Mapping & Bump Mapping


<table>
    <tr>
        <th>Before bump mapping</th>
        <th>After bump mapping</th>
    </tr>
    <tr>
        <th><img src="img/bump_mapping_before.png"/></th>
        <th><img src="img/bump_mapping_after.png"/></th>
    </tr>
</table>

### Microfact BSDF

To use various material, bsdfs that are more complicated than diffuse/specular are required. Here, we will first implement the classic microfacet BSDF to extend the capability of material in this pathtracer.

This pathtracer uses the Microfacet implementation basd on [pbrt].

Metallness = 1. Roughness 0 to 1 from left to right.

> Please note that the sphere used here is not an actual sphere but an icosphere. 

![Microfacet Demo](img/microfacet.png)

With texture mapping implemented, we can use `metallicRoughness` texture now. Luckily, `gltf` has a good support over metallic workflow.

![Metallic Workflow Demo](img/metallic_workflow.png)

### Environment Mapping


### Direct Lighting & MIS

> To stress the speed up of convergence in MIS, Russian-Roulette is disabled in this part's rendering.

> The tiny dark stripe is visible in some rendering result. This is because by default we do not allow double-sided lighting in this pathtracer.

> By default, number of light sample is set to 3.

<table>
    <tr>
        <th>Only sample bsdf 500spp</th>
        <th>Only sample light 500spp</th>
        <th>MIS 500spp</th>
    </tr>
    <tr>
        <th><img src="img/mis_bsdf_500spp.png"/></th>
        <th><img src="img/mis_light_500spp.png"/></th>
        <th><img src="img/mis_mis_500spp.png"/></th>
    </tr>
</table>

<table>
    <tr>
        <th>Without MIS 256spp</th>
        <th>With MIS 256spp</th>
    </tr>
    <tr>
        <th><img src="img/mis_without_mis_bunny_256spp.png"/></th>
        <th><img src="img/mis_with_mis_bunny_256spp.png"/></th>
    </tr>
    <tr>
        <th>Without MIS 5k spp</th>
        <th>With MIS 5k spp</th>
    </tr>
        <th><img src="img/mis_without_mis_bunny_5000spp.png"/></th>
        <th><img src="img/mis_with_mis_bunny_5000spp.png"/></th>
    </tr>
</table>




### Depth of Field

<table>
    <tr>
        <th>Depth of Field (Aperture=0.3)</th>
    </tr>
    <tr>
        <th><img src="img/depth_of_field.png"/></th>
    </tr>
</table>

### Future (If possible)

- [ ] Adaptive Sampling
- [ ] ReSTIR
- [ ] Refractive
- [ ] True B**S**DF (Add some subsurface scattering if possible?)
- [ ] Volume Rendering ~~(Ready for NeRF)~~

---


### History

- [x] Load mesh within arbitrary scene
    - [x] Triangle
    - [x] Integrate `tinygltf`
    - [x] Scene Node Tree

- [ ] Core
    - [x] G Buffer
    - [x] Russian Roulette
    - [x] Sort by material

- [ ] More BSDF
    - [x] Diffuse
    - [x] Emissive
    - [x] Microfacet
    - [ ] Reflective
    - [ ] Refractive
    - [ ] Disney

- [ ] BVH
    - [x] Basic BVH
        - [x] BoundingBox Array
        - [x] Construct BVH
        - [x] Traverse BVH
    - [x] Better Heuristics
        - [x] SAH 
    - [ ] MTBVH

- [ ] Texture
    - [x] Naive texture sampling
        - A Resource Mananger to help get the handle to texture?
    - [x] Bump mapping
    - [ ] Displacement mapping
    - [ ] Deal with antialiasing

- [ ] Better sampler
    - [ ] Encapsulate a sampler class
        - Gotta deal with cmake issue
    - [x] Monte carlo sampling
    - [x] Importance sampling
    - [x] Direct Lighting
    - [x] Multiple Importance Sampling

- [ ] Camera
    - [x] Jitter
    - [x] Field of depth
    - [ ] Motion blur 


- [ ] Denoiser 
    - [ ] Use Intel OpenImage Denoiser for now


### Log
09.20
- Basic raytracer
- Refactor integrator
- First triangle!

09.21-22
- Load arbitrary scene(only geom)
    - Triangle
    - ~~Primitive assemble phase~~(This will not work, see `README` of this commit)
    - Use tinygltf
        Remember to check [data type]((https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types)) before using accessor
    ![Alt text](img/accessor_data_types.png)
    - Done with loading a scene with node tree!
            ![blender_reference](img/blender_reference.png)
            ![rendered](img/first_scene.png)
        > Can't tell how excited I am! Now my raytracer is open to most of the scenes!
        - Scene with parenting relationship
            ![with_parenting](img/scene_with_parenting.png)

09.23-09.26
> Waste too much time on OOP. Eventually used C-style coding.

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


09.28

- SAH BVH(probably done...)

- Texture sampling
    - Try billboarding
    ![Alt text](img/billboarding.png)

09.29

- Texture mapping
    - Texture mapping test(only baseColor shading)
    ![Alt text](img/texture_mapping_test.png)
    - Bump mapping
        - Normals in world coordinate
    ![](img/bump_mapping_world_coord.png)
        - Before bump mapping
    ![](img/bump_mapping_before_annotated.png)
        - After bump mapping
    ![](img/bump_mapping_after_annotated.png)
    It might be a little difficult to notice the difference before and after bump mapping. **Please observe the logo on the box, more details are added.**


    - [ ] Texture aliasing is indeed quite serious!
        - [ ] However, to implement antialiasing for texture mapping, I may need to consider implementing mipmapping. 

- [ ] Microfacet model pbr failed...
    - [ ] Need to read through microfacet part and think about how to use roughness and metallic indeed

09.30

- [ ] Microfacet
    - [x] Metal Fresnel hack
    - [ ] Conductor
        - [ ] After mixing, need to consider how to sample
- [x] Camera 
    - [x] Antialiasing

10.1-10.2
Try to refactor camera
- Failed. gltf seems to have a really ambiguous definition of camera.

10.3
- [ ] Denoising
    - [x] OpenImage Denoiser built
        - CPU only for now
        - [ ] Figure out how to build `oidn` for cuda
    - [ ] Integrate it into project


10.4-10.6

- Microfacet

10.7 

- Environment map

10.8

- Fix random number issue(Maybe try to generate a better random number array in future?)
    - Before
![](img/before_fixing_random_number_issue.png)
    - After
![](img/after_fixing_random_number_issue.png)

    **Please notice the fracture on rabbit head before fixing**

10.9
- MIS (Finally!)

- Russian Roulette 
    - Pro: Speed up by 60%
    - Con: Lower the converge speed


- Depth of field
    - Add a realtime slider to adjust