CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Licheng CAO
  * [LinkedIn](https://www.linkedin.com/in/licheng-cao-6a523524b/)
* Tested on: Windows 10, i7-10870H @ 2.20GHz 32GB, GTX 3060 6009MB

Result
==============
<p align="center">
  <img src="img/cornell.2023-10-07_01-28-25z.5000samp.png"/>
</p>

Features
============
### Textures
|bump map| base map | result|
|:-----:|:-----:|:-----:|
|![cornell 2023-10-07_19-59-50z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/d0817e69-5939-4977-9019-f7b15b9bb4a9)|![cornell 2023-10-07_19-57-15z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/eeebc17e-e80a-442e-bb39-6d786f454d12)|![cornell 2023-10-07_20-01-36z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/1d9c0b60-5853-43e0-9b01-51a4ad7d1404)|

### Depth of field
|without DOF| with DOF |
|:-----:|:-----:|
|![cornell 2023-10-07_21-03-23z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/8f99d738-7b82-436d-b432-8619ac922788)|![cornell 2023-10-07_21-05-11z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/eb988061-bf35-40c6-ac31-df668a1022e8)|

### .obj loading and BVH tree
|Naive|BVH|
|:-----:|:-----:|
|![fpsNA](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/0eae0a65-57ee-43ff-b9b1-69bb1cf18256)|![fps](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/f3d4598b-c799-4bfe-b9da-65766349e6bd)|
* I used tinyobjloader to load mesh into the scene and implement a BVH to accelerate ray mesh intersection test.
  * I built the tree on CPU based on [PBR 4.3.1](https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies).
* For each ray bounce iteration, the ray will first test if it intersects with the bounding box in the scene. If the ray intersects the bounding box, it will continue to test if it hits the bounding box's child node until it find the neareset intersection with a triangle in the scene.
  * I used a stack to help me to implement the ray's behavior in an iterative way since GPU doesn't have good support for recusive bahaviors.
  * To accelerate the process, I added 2 if statements to get early termination of the intersection process. First, test if we already have a hit point that is nearer than the hit point we have on the current bouding box, if we do, we skip the current bounding box. Second, if we intersect a bounding box and want to intersect its children, we can decide the order to test children by the parent node's split axis. So that we can test the child that is near to the current ray first and have a great chance to skip the second child.
* And the fps is improved a lot:
<p align="center">
  <img src="https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/baaccda6-32e1-439e-9a94-09f9f3b51324"/>
</p>

### Cache first frame
|no cache|cache first frame|
|:-----:|:-----:|
|![cornell 2023-10-07_21-03-23z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/8f99d738-7b82-436d-b432-8619ac922788)|![cornell 2023-10-07_21-23-59z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/a25107d3-ce6b-4b7e-b5c6-304717ac1017)| 
* I also tried to cache first frame to improve fps. And the test result is shown as follow:
<p align="center">
  <img src="https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/9a7a6978-edc5-4db1-9e07-a57a53a8d4fd"/>
</p>
* 

### Reflection and Refraction
|reflection| refraction |
|:-----:|:-----:|
|![sphere 2023-10-07_21-19-10z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/a55be12d-3cea-4f83-bb7a-cd054beae718)|![refract](img/sphere.2023-10-06_22-33-00z.5000samp.png) |
* For my 
### Stream compaction
* Path continuation/termination
  * I used thrust::copy_if and thrust::remove_if to remove paths that hit light or nothing and record them in a output array, so that the path tracer will have less path to check after each bounce and result in a better performance.
* Sort materials
  * I tried to sort the paths based on the material of intersections with thrust::sort_by_key, but it actually resulted in a lower fps. The result may be that the cost of sorting overweighed the performance improvement it brings to my path tracer.



Third Party
============
* Code
  * [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
  * [imgui](https://github.com/ocornut/imgui)
  * [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h)
* Assets
  * [bunny](https://github.com/harzival/stanford_bunny_obj_mtl_jpg/blob/master/bunny.obj)
  * [planet](https://www.turbosquid.com/3d-models/3d-stylized-planet-system-4k-free-1973128)


