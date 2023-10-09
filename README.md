CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

Han Wang

Tested on: Windows 11, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz 22GB, GTX 3070 Laptop GPU

# Overview
![Unlock FPS](img/cornell.2023-10-09_16-26-09z.2719samp.png)


This project is a Monte Carlo path tracer run on GPU based on CUDA. It is a rendering algorithm that simulates the behavior of light in a scene by tracing rays from the camera into the scene. Using a GPU and CUDA for Monte Carlo path tracing allows for massive parallelization, speeding up the ray tracing and light simulation steps. CUDA is a parallel computing platform and application programming interface (API) model that efficiently utilizes the GPU's parallel processing capabilities.




# Render features:

### 1. Refraction (e.g. glass/water)
|reflect|reflect+refrect(glass)|difffuse
|:-----:|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_13-52-42z.1734samp.png" width="200" height="200">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_09-07-59z.1028samp.png" width="200" height="200">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_14-13-55z.1091samp.png" width="200" height="200">

In this part, I made three materials including reflected metal,  reflect+refrect glass, and normal diffuse material used for comparison. I tried to make the refract and reflect portion getting close to the real world's glass and successfully built my glass ball. It is easy to see that there are lighted areas below the glass ball showing that the light passed through the glass ball. In contrast, I also put the same ball but with the diffusion surface, the area below the ball is shadowed and dark.




### 2. Physically-based depth-of-field
|with DOF|without DOF|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_13-31-12z.2063samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_14-22-16z.1047samp.png" width="300" height="300">

In this part, I write an interesting camera effect called depth of field. It refers to the range within a scene that appears in sharp focus, creating a clear and visually pleasing image. The DOF effect is achieved by controlling the focal length of the camera lens and the aperture size, determining how much of the scene is in focus and how much is blurred. This technique allows for artistic control over the perception of depth, emphasizing certain subjects while softening the background or foreground, enhancing the overall visual impact of the image or video.





### 3. Stochastic Sampled Antialiasing
|with Antialiasing|without Antialiasing|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_14-13-55z.1091samp.png" width="500" height="500">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_15-21-00z.4090samp.png" width="500" height="500">

From the image comparison, we can see that at the edge of the sphere, there is a great difference between antialiasing and no antialiasing. The edge with antialiasing is greatly smoother than without it. The operation is also simple, we just apply a small random float number between the iterations of the path. Because we basically need average between iterations, the average step in antialiasing is already completed.


### 4. Subsurface scattering
|Subsurface scattering|diffuse surface|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_08-47-59z.1063samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_08-58-21z.944samp.png" width="300" height="300">


### 5. defining object motion, and motion blur
|motion blur direction 1|motion blur direction 2|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_14-34-10z.547samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-10-09_15-00-34z.637samp.png" width="300" height="300">


# Analysis

### Stream Compaction


### Material Sorting




### Open Space & close space
|in the box|without the box|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/compare1.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/compare2.png" width="300" height="300">

Based on the image, we can easily see that we need an average of 634.047 MS to generate an iter in the box while we only need 104.308 MS to generate an iter without the box. It's obvious that we need less time to iter without a box. The reason is also obvious: There is less light bouncing in the open space compared to the inside box, thus the iteration terminates early and needs fewer resources. 
# reference

https://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models#ProjectiveCamera::focalDistance
https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#ConcentricSampleDisk
https://en.wikipedia.org/wiki/Schlick%27s_approximation
561 HW6 (personal reference)
