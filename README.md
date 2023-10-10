CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xiaoxiao Zou
  * [LinkedIn](https://www.linkedin.com/in/xiaoxiao-zou-23482a1b9/)
* Tested on: Windows 11, AMD Ryzen 9 7940HS @ 4.00 GHz, RTX 4060 Laptop 

### Features

Visual:
* Anti-aliasing - :two: points
* Mirror - :zero: points
* Refraction - :two: points (glass with specular component)
* Plastic Material :?: points (mixture of diffuse, refraction, specular, material)
* Phong / Blinn-Phong with cosine lobine importance sampling - :?: points 
* Microfacet shading with Blinn-Phong - :?: points 
* Texture/Normal Mapping with mesh loading - :six: points
* Smooth Shading :?: points 

Mesh: 
* Obj/Mtl file loading - :two: points

Performance:
* First Bounce Cache - :zero: points
* Stream Compaction - :zero: points
* Material Sort - :zero: points
* BVH - :six: points

![sponza](./img/sponza.png)

Note: some of the features are real time adjustable by just click on some keys. 

#### First Bounce Cache (C-enabled, V-Disabled)
This speeds up the performance for most situations. 

#### Antialiasing (A-enabled, Q-Disabled)
This feature does not give apparently visual improvement since the ray is scattered out on diffuse material, after certain amount of sampling, it gets smooth shaded. Since for antialiasing, the first bounce is not always same, we cannot cach the first bounce anyway. In my program, once the first bounce is on, antialiasing will also be turned off. 

#### Material Sort (M-enabled, N-Disabled)
This feature actually slow down the process when number of materials and geometry are relatively small.


#### Materials & Shading (0-Phong, 1-BlinnPhong, 2-BlinnPhong with Microfacet)

* refraction

* specular highlight (importance sampling with cosine lobe) or Phong shading

This method gives a cosine lobe sampling at reflected direction, which takes in a extra specular exponent (Ns in mtl, exponent in txt, range from 0-1000) to determine how scattered the ray. 

* mixture of multiple materials

This method gives possibility to generate various types of materials, white coated material, colored glass ball with reflection, and etc. 

* Blinn-Phong

Sampling at reflected angles will cost problems at gazing angles due to reflected rays close to the surface. Then blinn phong can sample at half vector. 

* Blinn-Phong with Microfacet

Microfacet gives some geometrical property to the surface since in real world, most plane is not perfectly flatten. 


#### Obj/Mtl file & texture loading 

Reading files using tinyobj and stb_image. 

#### Bounding Volume Hierarchy

