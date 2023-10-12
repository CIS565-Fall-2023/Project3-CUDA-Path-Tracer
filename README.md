CUDA Path Tracer
================
![Path Tracer](/img/bunny.png)

Jason Xie, Fall 2023

[LinkedIn](https://www.linkedin.com/in/jia-chun-xie/)
[Website](https://www.jchunx.dev)

## Overview

A path tracer written in CUDA.

In this project, I take advantage of the massively parallel processing power of GPU to render photorealisitc images. The end result is a interactive path tracer featuring specular and diffuse shading, along with several optimizations to improve performance.

## Features

### ✨ Diffuse / Specular shading

Simple BSDF shading for diffuse and specular surfaces.

| 100% Diffuse |
|-----------------|
| ![Diffuse](/img/pig_diffuse.png) |

| 100% Specular |
|-----------------|
| ![Specular](/img/pig_specular.png) |


### 🙅‍♂️ Path termination via stream compaction

When rays hit a light source or exceed the maximum number of bounces, they should no longer be traced. Stream compaction is implemented in this project using `thrust::partition`, where valid rays are moved to the front of the array and invalid rays are moved to the back. The number of valid rays is then returned.

Let's compare the number of active paths for an open Cornell box vs a closed Cornell box.

| Open Cornell Box | Closed Cornell Box |
|-----------------|-----------------|
| ![Open Cornell Box](/img/cornell_open.png) | ![Closed Cornell Box](/img/cornell_closed.png) |

![plot](/img/path_compare.png)

As expected, the number of active paths for the open Cornell box is significantly higher than the closed Cornell box. This is because rays can escape the open Cornell box and terminate early, while rays in the closed Cornell box are forced to bounce around until they exceed the maximum number of bounces.

What about actual performance? 

| Path termination on `cornell.txt` | ms / frame |
|-----------------|------------|
| No               |  13       |
| Yes              |  27        |

The current form of path termination is slower than the naive approach, which suggests that the overhead of stream compaction is greater than the performance gains from terminating paths early for this simple scene.

### ⚡️ Continuous memory rays via sorting

To prevent inefficient memory access patterns, we perform memory coalescing by sorting rays by their material type. This ensures that rays with the same material type are stored contiguously in memory.

Performance Comparison:

| Sorting on `cornell.txt` | ms / frame |
|-----------------|------------|
| No               |  27       |
| Yes              |  54        |

Sorting rays by material type is slower than the naive approach, which suggests that the overhead of sorting rays is greater than the performance gains from memory coalescing.

### 🏦 First bounce caching

In addition to sorting rays by material type, we also cache the first bounce of each ray. This allows us to avoid having to recalculate the first bounce for each ray in the scene. It is possible to perform this optimization because the first intersection is always deterministic.

Performance Comparison:

| Max Bounces | With Caching (ms/ frame) | Without Caching (ms / frame) |
|-----------------|------------|------------|
| 2               |  13.5       |  13.5       |
| 3              |  13.7       |  13.7       |
| 4              |  14.0       |  14.0       |
| 5              |  25.0       |  22.0       |
| 6              |  27.0       |  26.6       |
| 7              |  27.5       |  26.8       |
| 8              |  26.6       |  27.0       |

![bounce plot](/img/first_bounce.png)

It appears that storing the first bounce provides minor improvements to the frame time, and performance gains remains constant over the number of bounces. This can be attributed to the fact that the savings from caching is independent of the number of subsequent bounces.

### 🐇 GLTF mesh loading

In order to push the limits of the path tracer, I implemented a GLTF mesh loader to load complex geometries. First, the GLTF file is parsed using tinygltf. Then, mesh vertices and indices are loaded from raw bytes into CUDA device memory.

### 🐙 Hierarchical spatial data structures

In the naive path tracing approach, a ray must check with every object and triangle in the scene to find the closest intersection. This is inefficient, especially when the scene contains thousands of objects. To improve performance, we can partition the scene using octrees🐙. octrees🐙 significantly reduce the number of intersection tests required by eliminating large swatch of the scene that do not intersect with the ray.

![illustration](/img/octree_illustration.png)

To implement octrees🐙, an axis-aligned bounding box is first created for the entire mesh object. Then, the bounding box is recursively subdivided into 8 octants until each octant contains at most N triangles or the maximum depth is reached. Triangles spanning across octants are accounted for by duplicating them across octants. To check for an intersection, the ray first checks for a collision with the bounding box. If a collision is found, the ray recursively checks for a collision with child nodes until a leaf node is reached. Then, the ray checks for a collision with each triangle in the leaf node.


Performance Comparison

`🐇.gltf: Triangles: 30.3k, Vertices: 15.3k` 
| Octree Depth | ms / frame | FPS |
|--------------|------------|-----|
| 0            | 383        | 2.6 |
| 1            |  378        | 2.6 |
| 3            |  429        | 2.3 |
| 16            |  429        | 2.3 |
| 64            |  429        | 2.3 |

For this scene, performance improves slightly at low depths, but quickly saturates. More complex scenes and better memory access patterns may help push performance further.

## Acknowledgements

1. [tinygltf](https://github.com/syoyo/tinygltf) for GLTF parsing
2. [Stanford Bunny](https://sketchfab.com/3d-models/stanford-bunny-df325cfe1fa24108bdfef58ba7e88b3a)
3. [Minecraft Pig](https://sketchfab.com/3d-models/minecraft-pig-98b5949506b84484aecacea589f38c45)

## Bloopers

|Missing triangles due to not accounting for triangles that span across octree🐙 nodes.|
|-----------------|
![bad octree](/img/bad_octree1.png)


|Attempt to optimize using CUDA streams, but kernels were accessing the same memory chunk.|
|-----------------|
![bad stream](/img/bad_stream.png)