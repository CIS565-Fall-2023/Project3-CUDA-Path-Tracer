CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Jason Li   ([LinkedIn](https://linkedin.com/in/jeylii))
* Tested on: Windows 10, Ryzen 5 3600X @ 3.80GHz 32 GB, NVIDIA RTX 4070 

# **Summary**
This project is a GPU Path Tracer implemented using C++ and CUDA. It is able to load GLTF files and stores & accesses the data for such files in a BVH (Bounding Voume Hierarchy) data structure.
 
# **Features**
## **Basic Path Tracing using BSDFs**
The first feature to implement was shading the scene using path tracing instead of the naive shading method. For the scattering of light, three BSDF (Bidirectional Scattering Distribution Functions) were implemented - ideal diffuse shading, perfect specular shading, and a mix of the two. Below are some comparisons between the BSDFs mentioned.

### **Stream Compaction**
One optimization that was made was using stream compaction to terminate paths before their set trace depth was reached. The paths are compacted once per iteration; after each path's color is accumulated. This is implemented using the `thrust::partition()` function from NVIDIA's Thrust CUDA Library. Below is a performance analysis of the inclusion of stream compaction.
### Compaction Performance Analysis


### **Sorting Paths by Material**
Another optimization that was implemented was making paths contiguous in memory by material type during each iteration. 
This occurrs once per iteration, once the intersections are computed and each path is associated with a material type. This is implemented using the `thrust::sort_by_key()` function from NVIDIA's Thrust CUDA Library. Below is a perofmance analysis of the inclusion of this optimization.
### Material Sorting Performance Analysis 

### **First Bounce Caching**
The last basic optimization made was caching the intersections from the first path bounce (directly after the leave the camera) to be used across all subsequent iterations of path tracing. This occurrs in the first iteration right after intersection computation. This is implemented by copying the information about the first bounces into a designated memory location on the GPU and copying the information back from that memory location into each following iteration's intersection buffer using `cudaMemcpy()`. Below is a performance analysis of the inclusion of this optimization.
### First Bounce Caching Performance Analysis


## **glTF File Mesh Loading & Rendering**
One advanced feature that was added was the loading and rendering of arbitrary meshes from one *or multiple* glTF files. To do this, a "GLTF" entry can be included in the `.txt` file used to load the scene for each glTF file to be loaded; the path and filename of the `.gltf` file must be included on the next line. The meshes included in these files were parsed and read using the `tinygltf` library, which can be found here. This functionality is currently limited to meshes without textures; only meshes purely using RGB values to color the scene can be loaded.


## **Hierarchical Spatial Data Structure - BVH**
One advanced feature that was 
This greatly helped with the performance of rendering the much more advanced meshes loaded in through 

https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

https://github.com/syoyo/tinygltf

Piplup	https://skfb.ly/6Vt7x

Trees	https://skfb.ly/6BOWV

Water Bottle	https://skfb.ly/o7CKp

McCree	https://skfb.ly/6sBDo

Bulbasaur	https://skfb.ly/6SZ9B
