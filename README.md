CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

- [ ] Load Mesh
    - [ ] Primitive
        - [x] Triangle
    - [ ] Primitive assemble phase(How to transport mesh loaded on host to device?)
        This is troublesome because we need to consider OOP design on device side.
        We cannot simply use `cudaMemcpy` now as we may need to malloc a pointer array on device to use OOP. The pointer will also need a piece of memory to point at. We cannot `cudaMemcpy` the memory directly as that would involve accessing device memory on the CPU side, which is not allowed. 
        *A possible solution*: We can assemble the array of different primitives on the CPU side: `vector<Triangle>`, `vector<Sphere>`, etc. Then when loading occured, we will load each primitives by their type. In this way we can also have a more tighted device memory.
        > This is not working! One lesson learnt: Remember to always new objects on device! Otherwise, the virtual functions of the copied child class instances will point to an address on the host memory, leading to a illegal memory access. 

        Pseudo code:
        ```cpp
        class Primitive;
        class Triangle; //:public Primitive;
        class Sphere; //:public Primitive;

        // CPU side
        class Model{
            std::vector<Triangle> triangles;
            std::vector<Sphere> spheres;
            int getPrimitiveSize(){
                return triangles.size() + spheres.size() // + other extra primitves...
            }
        };

        // Loading phase
        Primitive ** primitives;
        Triangle * dev_triangles;
        Sphere * dev_spheres;
        void load(){
            int primitive_offset = 0;
            cudaMalloc(primitives, model->getPrimitiveSize() * sizeof(Primitive));

            cudaMalloc(dev_triangles, model->triangles.size() * sizeof(Triangle));
            cudaMemcpy(dev_triangles, model->triangles.data(), cudaMemcpyKind::HostToDevice);

            
            cudaMalloc(dev_spheres, model->spheres.size() * sizeof(Sphere));
            cudaMemcpy(dev_spheres, model->spheres.data(), cudaMemcpyKind::HostToDevice);

            dim3 loadPrimitiveBlocks(model->triangles.size() + (blockSize-1)/blockSize);

        }

        template<typename T>
        __global__ void copyPointer(int offset, int size, Primitive ** dev_primitives, T *dev_sub_primitives){
            int index /* TODO: Calculate it */;
            if (index < size) {
                dev_primitives[offset + index] = dev_sub_primitives + index;
            }
        }
        ```
        - [ ] Use tinygltf
        ![Alt text](img/accessor_data_types.png)
        [Link](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types)
            - [x] Done with loading a scene with node tree!
                ![blender_reference](img/blender_reference.png)
                ![rendered](img/first_scene.png)
                Can't tell how excited I am! Now my raytracer is open to most of the scenes!
                - Scene with parenting relationship
                ![with_parenting](img/scene_with_parenting.png)
