#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include <string>
class CudaTexture {
private:
	cudaArray* m_cuArray = nullptr;
	cudaTextureObject_t m_texObj = 0;
public:
	int m_width;
	int m_height;
	CudaTexture();
	~CudaTexture();
	void LoadTexture(const char* path);
	void LoadTexture(const std::string& path);
	void DeloadTexture();
};
