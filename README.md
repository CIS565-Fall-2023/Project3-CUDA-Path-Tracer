CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Helena Zhang
* Tested on: Windows 11, i7-10750 @ 2.6GHz 16GB, Geforce RTX 2060 6GB

Analysis
================
* stream compaction
* cache first hit
* material sort
Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.

Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels.
  
Extra Features
================
**Mesh import (gltf)**
insert images
performance impact: tbd, compare # vertices vs performance

**Direct Lighting**
insert low iteration low depth images
performance impact: tbd

**Refraction**
insert squid image without glass & water and with glass & water
performance impact: tbd

**Depth of Field Camera**
Lens Radius Effect: (insert 2 sheep pictures)
Focus Distance Effect: (insert 3 sheep pictures)
Performance impact: none



