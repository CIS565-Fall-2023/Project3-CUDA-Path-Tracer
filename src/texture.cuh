#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
#include <cstdio>
#include <memory>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
class CudaTexture {
private:
	CudaTexture(const CudaTexture& other);
	CudaTexture& operator=(const CudaTexture& other);
	void LoadTexture(const char* path);
	void DeloadTexture();
public:
	cudaArray* m_cuArray = nullptr;
	cudaTextureObject_t m_texObj = 0;
	int m_width;
	int m_height;
	CudaTexture(const char* path);
	CudaTexture(const std::string& path);
	CudaTexture(CudaTexture&& other);
	~CudaTexture();
};
