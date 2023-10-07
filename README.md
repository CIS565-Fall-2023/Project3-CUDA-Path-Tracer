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
One advanced feature that was added was the loading and rendering of arbitrary meshes from glTF files. 


## **Hierarchical Spatial Data Structure - BVH**

