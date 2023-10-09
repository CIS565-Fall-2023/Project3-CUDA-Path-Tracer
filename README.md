CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

Han Wang

Tested on: Windows 11, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz 22GB, GTX 3070 Laptop GPU

# Overview
![Unlock FPS](img/cornell.2023-09-29_13-31-12z.1734samp.png)
This project is a Monte Carlo path tracer run on GPU based on CUDA. It is a rendering algorithm that simulates the behavior of light in a scene by tracing rays from the camera into the scene. Using a GPU and CUDA for Monte Carlo path tracing allows for massive parallelization, speeding up the ray tracing and light simulation steps. CUDA is a parallel computing platform and application programming interface (API) model that enables efficient utilization of the GPU's parallel processing capabilities.




# Render features:

### 1. Refraction (e.g. glass/water)
|reflect|reflect+refrect|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_09-07-59z.1028samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_09-07-59z.1028samp.png" width="300" height="300">

### 2. Physically-based depth-of-field
|with DOF|without DOF|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_13-31-12z.2063samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_13-31-12z.2063samp.png" width="300" height="300">


### 3. Stochastic Sampled Antialiasing
|with Antialiasing|without Antialiasing|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_13-31-12z.2063samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_13-31-12z.2063samp.png" width="300" height="300">


### 4. Subsurface scattering
|Subsurface scattering|diffuse surface|
|:-----:|:-----:|
|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_08-47-59z.1063samp.png" width="300" height="300">|<img src="https://github.com/Ibm510000/Project3-CUDA-Path-Tracer/blob/main/img/cornell.2023-09-29_08-58-21z.944samp.png" width="300" height="300">


### 5. defining object motion, and motion blur

