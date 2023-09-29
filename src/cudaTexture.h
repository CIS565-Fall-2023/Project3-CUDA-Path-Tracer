#pragma once

#include "common.h"

class Texture2D
{
public:
	Texture2D();
	Texture2D(const std::string& img_path, bool flip_v = false);
	~Texture2D();

	void Create(const std::string& img_path, bool flip_v = false);
	void Free();
public:
	float* m_RawImg;
	int m_Width, m_Height;
	cudaArray_t m_TexArray;
	cudaTextureObject_t m_TexObj;
};

class CudaTexture2D
{
public:
	CPU_ONLY CudaTexture2D(const cudaTextureObject_t& tex_obj = 0);
	GPU_ONLY float4 Get(const float& x, const float& y) const;

public:
	cudaTextureObject_t  m_TexObj;
};