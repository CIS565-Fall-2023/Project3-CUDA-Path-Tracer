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

* [texture.cuh](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/blob/main/src/texture.cuh) and [texture.cu](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/blob/main/src/texture.cu) are used to handle textures with CUDA. I applied bump mapping and texture mapping to help me render meshes. And other textures like roughness maps can also be loaded to help render meshes as long as the shading parts supports that material attribute.

|bump map| base map | result|
|:-----:|:-----:|:-----:|
|![cornell 2023-10-07_19-59-50z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/d0817e69-5939-4977-9019-f7b15b9bb4a9)|![cornell 2023-10-07_19-57-15z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/eeebc17e-e80a-442e-bb39-6d786f454d12)|![cornell 2023-10-07_20-01-36z 15000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/1d9c0b60-5853-43e0-9b01-51a4ad7d1404)|


### Depth of field
* To achieve depth of field (DOF) effects, I referred to [PBRT 6.2.3](https://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#TheThinLensModelandDepthofField) and successfully implemented the thin lens model. This implementation significantly contributes to generating high-quality output images for the ray tracer.

|without DOF| with DOF |
|:-----:|:-----:|
|![cornell 2023-10-07_21-03-23z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/8f99d738-7b82-436d-b432-8619ac922788)|![cornell 2023-10-07_21-05-11z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/eb988061-bf35-40c6-ac31-df668a1022e8)|


### .obj loading and BVH tree
* I employed the tinyobjloader library to efficiently load meshes into the scene. Additionally, I implemented a Bounding Volume Hierarchy (BVH) to optimize ray-mesh intersection tests, thereby enhancing the rendering performance.
  * The BVH tree is built on CPU based on [PBRT 4.3.1](https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies).
* During each iteration of ray bouncing, the ray undergoes a series of tests. Initially, it checks for intersections with the bounding boxes in the scene. If an intersection with a bounding box occurs, the ray proceeds to check for intersections with the child nodes of that bounding box. This process continues until the ray finds the nearest intersection with a triangle in the scene, ensuring accurate rendering of the scene's geometry.
  * I utilized a stack data structure to help me implement the ray's behavior in an iterative way since GPUs don't have good support for recusive bahaviors.
  * To expedite the intersection process, I introduced two crucial optimizations. First, I implemented an early termination mechanism by checking if we already have a hit point closer than the one present within the current bounding box. If such a hit point exists, we skip further examination of the current bounding box. Second, I strategically determined the order in which to test the children of a bounding box based on the parent node's split axis. This allows us to prioritize testing the child that is closer to the current ray's position, increasing the likelihood of early termination and further improving the ray tracing efficiency.
* These optimizations have significantly improved the frames per second (FPS), making the rendering process much more efficient:
<p align="center">
  <img src="https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/baaccda6-32e1-439e-9a94-09f9f3b51324"/>
</p>

|Naive|BVH|
|:-----:|:-----:|
|![fpsNA](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/0eae0a65-57ee-43ff-b9b1-69bb1cf18256)|![fps](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/f3d4598b-c799-4bfe-b9da-65766349e6bd)|


### Cache first frame
* Cache first frame feature is implemented to improve fps. And the test result is shown as follow:
<p align="center">
  <img src="https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/9a7a6978-edc5-4db1-9e07-a57a53a8d4fd"/>
</p>

* The result shows that caching first frame will slightly improve fps, especially with lower ray bounce. However, it will also introduce some artifacts if the ray is jittered by some random number for anti-aliasing as shown:

|no cache|cache first frame|
|:-----:|:-----:|
|![cornell 2023-10-07_21-03-23z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/8f99d738-7b82-436d-b432-8619ac922788)|![cornell 2023-10-07_21-23-59z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/a25107d3-ce6b-4b7e-b5c6-304717ac1017)| 


### Reflection and Refraction
* Implementing basic reflection in ray tracing is quite straightforward. However, it's crucial to handle the Lambertian reflection model carefully.
* For the refraction, I referred [here](https://zhuanlan.zhihu.com/p/303168568) for the Schlick's approximation and the accurate fresnel evaluation and use [glm::refract](https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml) to calculate the ray direction.

|reflection| refraction |
|:-----:|:-----:|
|![sphere 2023-10-07_21-19-10z 5000samp](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/a55be12d-3cea-4f83-bb7a-cd054beae718)|![refract](img/sphere.2023-10-06_22-33-00z.5000samp.png) |

### Stream compaction
* Path continuation/termination
  * I used thrust::copy_if and thrust::remove_if to remove paths that hit light or nothing and record them in a output array, so that the path tracer will have less path to check after each bounce and early terminations, which results in a higher fps:
 
|no compaction| compaction |
|:-----:|:-----:|
|![n捕获](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/bf22d5ad-94c0-431c-9d0d-a5053177a30a)|![捕获](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/934e70ff-5ad1-4d02-ba06-48afe956e39d)|

* Sort materials
  * I tried to sort the paths based on the material of intersections with thrust::sort_by_key, but it actually resulted in a lower fps. The reason may be that the cost of sorting overweighed the performance improvement it brings to my path tracer.

|without sort| sorted |
|:-----:|:-----:|
|![捕获](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/9fc79cc6-0270-45b1-867d-52317eae8bd9)|![n捕获](https://github.com/LichengCAO/Project3-CUDA-Path-Tracer/assets/81556019/575bc562-5fe9-4cdb-9f5e-cc5ef392c732)|

Third Party
============
* Code
  * [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
  * [imgui](https://github.com/ocornut/imgui)
  * [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h)
* Assets
  * [bunny](https://github.com/harzival/stanford_bunny_obj_mtl_jpg/blob/master/bunny.obj)
  * [planet](https://www.turbosquid.com/3d-models/3d-stylized-planet-system-4k-free-1973128)


