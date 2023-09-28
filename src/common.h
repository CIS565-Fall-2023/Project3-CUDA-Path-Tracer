#pragma once

#include <cstdio>
#include <memory>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>

#define CPU_ONLY __host__
#define GPU_ONLY __device__
#define CPU_GPU __device__ __host__
#define INILNE __forceinline

#define ERRORCHECK 1
#define SAH_BVH 1

GPU_ONLY static constexpr float Pi				= 3.1415927f;
GPU_ONLY static constexpr float TWO_Pi			= 6.2831853f;
GPU_ONLY static constexpr float InvPi			= 0.3183099f;
GPU_ONLY static constexpr float Inv2Pi			= 0.1591549f;
GPU_ONLY static constexpr float Inv4Pi			= 0.0795775f;
GPU_ONLY static constexpr float PiOver2			= 1.5707963f;
GPU_ONLY static constexpr float PiOver4			= 0.7853981f;
GPU_ONLY static constexpr float Sqrt2			= 1.4142136f;
GPU_ONLY static constexpr float Epsilon			= 0.0001000f;
GPU_ONLY static constexpr float Sqrt_One_Thrid	= 0.5773502f;
GPU_ONLY static constexpr float Float_MAX		= 1000000.f;
GPU_ONLY static constexpr float Float_MIN		= -1000000.f;

#define SafeCudaFree(ptr) if(ptr) cudaFree(ptr); ptr = nullptr;
#define SafeCudaFreeArray(arr_ptr) if(arr_ptr) cudaFreeArray(arr_ptr); arr_ptr = nullptr;

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#if ERRORCHECK
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#else 
#define checkCUDAError(msg)
#endif
void checkCUDAErrorFn(const char* msg, const char* file, int line);

#define uPtr std::unique_ptr
#define sPtr std::shared_ptr
#define wPtr std::weak_ptr

#define mkU std::make_unique
#define mkS std::make_shared