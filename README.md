# CUDA Path Tracer

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Aditya Gupta
  * [Website](http://adityag1.com/), [GitHub](https://github.com/AdityaGupta1), [LinkedIn](https://www.linkedin.com/in/aditya-gupta1/), [3D renders](https://www.instagram.com/sdojhaus/)
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, NVIDIA GeForce RTX 2070 8GB (personal laptop)
  * Compute capability: 7.5

### Features

- Basic path tracer with lambert and specular reflection
- Toggleable option to sort rays by material type
- Toggleable first bounce cache
  - Disabled for now because it seems to slow performance at higher maximum depths
- Russian roulette path termination
- Specular refraction
- Depth of field
- glTF mesh loading
- BVH construction and traversal

### Planned features

- Add slider for depth of field? (which should reset pathtracer state when changed)
- Open Image Denoise
- Direct/full lighting?