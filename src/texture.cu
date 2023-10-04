#include "texture.cuh"
#include "stb_image.h"
#include <iostream>
#include "glm/glm.hpp"

CudaTexture::CudaTexture(const char* path)
{
	LoadTexture(path);
}

CudaTexture::CudaTexture(const std::string& path)
	:CudaTexture(path.c_str())
{}

CudaTexture::CudaTexture(CudaTexture && other)
	:m_cuArray(other.m_cuArray),m_height(other.m_height),m_width(other.m_width),m_texObj(other.m_texObj)
{
	std::cout << "cuda texture move constructor " << std::endl;
	other.m_cuArray = nullptr;
	other.m_texObj = 0;
}

CudaTexture::~CudaTexture()
{
	DeloadTexture();
}

void CudaTexture::LoadTexture(const char* path)
{
	if (path != "") {
		int nrChannels;
		stbi_set_flip_vertically_on_load(true);
		std::cout << "load texture from: " << path << std::endl;
		auto data = stbi_loadf(path, &m_width, &m_height, &nrChannels, 4);
		if (data) {
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
			cudaMallocArray(&m_cuArray, &channelDesc, m_width, m_height);
			const size_t spitch = m_width * sizeof(float4);
			// Copy data located at address h_data in host memory to device memory
			cudaMemcpy2DToArray(m_cuArray, 0, 0, data, spitch, spitch, m_height, cudaMemcpyHostToDevice);

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			

			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = m_cuArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			cudaCreateTextureObject(&m_texObj, &resDesc, &texDesc, NULL);
		}
		else {
			std::cout << "Failed to load texture" << std::endl;
		}
		stbi_image_free(data);
	}
	else {
		std::cout << "invalid image file" << std::endl;
	}
}

void CudaTexture::DeloadTexture()
{
	if (m_texObj != 0) {
		std::cout << "destroy cuda texture object" << std::endl;
		cudaDestroyTextureObject(m_texObj);
		m_texObj = 0;
	}
	if (m_cuArray != nullptr) {
		std::cout << "destroy cuda texture array" << std::endl;
		cudaFreeArray(m_cuArray);
		m_cuArray = nullptr;
	}
}
