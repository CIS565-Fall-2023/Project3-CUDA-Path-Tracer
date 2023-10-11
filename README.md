CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Tianyi Xiao
  * [LinkedIn](https://www.linkedin.com/in/tianyi-xiao-20268524a/), [personal website](https://jackxty.github.io/), [Github](https://github.com/JackXTY).
* Tested on: Windows 11, i9-12900H @ 2.50GHz 16GB, Nvidia Geforce RTX 3070 Ti 8032MB (Personal Laptop)

### Render Images
![](/img/wahoo_1.png)
![](/img/refraction_1.png)

### Stream Compaction

With stream compaction, we could terminate the rays that reach end, like reach outside of the scene, or hit the light.

![](/img/graphs/Stream%20Compation%20ray%20decrease.png)
Above graph record how rays are terminated with stream compaction, using the basic cornell box scene below, within one render iteration.
![](/img/jitterRay_cmp.png)
When it comes to the 7th bounce, there are just about 1/6 rays left compared with the beginning stage.

obj box scene:
![](/img/cornellDode.png)
closed box scene:
![](/img/closed_box.png)
Statistics of render time of each frame in different scenes (SC == Stream Compaction): 
![](/img/graphs/Stream%20Compaction%20scenes.png)

Then, I test the average render time of one frame in different scenes. The first scene is just the basic conrell box scene. The second and the third scene are both the obj box scene above, which combine the cornell box scene with several objects, so it's more complicated. But for the third scene, rays bounce 20 times instead of 8 times. (Other three scenes all bounce only 8 times, which is enough for render.)

From these three scenes, we can see that the stream compaction could speed up the render process. When the scene is simple, the speed up isn't obvious. But when tested in complicated scene, the positive effect of stream compaction is clearer. Also, as the maximum bounce times increase, the stream compact works better, since with more and more ray terminated, the render speed in succeeding bounce is higher and higher.

The forth scenes is closed box scene, which means no ray would reach outside of this scene. Therefore there will be a lot fewer rays terminated after each bounce. And from the result we can see that, in this situation, the stream compaction actually slow down the render speed. Because there are not many ray terminated in each bounce, the cost of stream compaction itself is higher than the time decrease of shading process.

### Material Sort
refraction scene:
![](/img/Refraction.png)

Statistics of render time of each frame in different scenes (MS == Material Sort):
![](/img/graphs/Material%20Sort.png)

The new refraction box scene is shown above. Actually the material sort slow down the render process. I think the reason is that, the cost of material sort is higher than the efficiency it brings. And there are only at most several branches relative to material kinds in shading stage. Maybe in scenes with fewer pixels and a with a lot more various kinds of materials supported (like different shading methods, Phong, PBR, etc.), the material sort could work out.

### First Bounce Cache
wahoo scene:
![](/img/wahoo_0.png)
Statistics of render time of each frame in different scenes:
![](/img/graphs/First%20Bounce%20Cache.png)

Caching the first bounce result could increase the render speed a little, without doubt. Though the speed up is not quite obvious, anyway it's still helpful.

### BVH Tree

The BVH part cost most of my time(I think nearly 60%), in which I face lots of bugs and problems (and luckily fix them all, finally). In conclusion, there are two toughest problems which I'd like to share about.

The first one is about the box intersection judgement. Initially I first check if the ray go through any pair of bounding box faces, then I check if the ray origin point is inside the box.

The second one is about the the tree division. I build bvh tree from the top to the bottom, and to divide objects in one parent bounding box to two children bounding box, initially I divide them accoding to its position compared with bounding box central position. However it makes the tree highly unbalanced. Now I sort the object of their position in specific axis, and deviced them into two equal parts according to the sort result.

many object scene
![](/img/manyObj.png)
Statistics of render time of each frame in different scenes:
![](/img/graphs/BVH.png)

These two main update (and also many trivial modificatoin together) make my BVH tree really effcient now. From the bar graph result above, we can see that the BVH tree increase the render speed dramatically, as we expect. With BVH tree, we could sucessfully limit the render time of complicated scene within a resonable limit.

### Render Features

This project also support some render features.

**Obj Load**

![](/img/manyObj.png)

**Microfacet**

![](/img/microfacet.png)

**Refraction**

![](/img/refraction_0.png)

**Jitter Ray**

jitter ratio is 1e-3
![](/img/jitterRay_1e_3_8d.png)
jitter ratio is 1e-2, the result is more blur
![](/img/jitterRay_1e_2_8d.png)

**Depth Of Field**

![](/img/Depth_Of_Field.png)