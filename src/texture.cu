#include "texture.cuh"
#include "stb_image.h"
#include <iostream>
#include "glm/glm.hpp"
CudaTexture::CudaTexture()
{}

CudaTexture::~CudaTexture()
{
	DeloadTexture();
}

void CudaTexture::LoadTexture(const char* path)
{
	DeloadTexture();
	if (path != "") {
		int nrChannels;
		stbi_set_flip_vertically_on_load(true);
		std::cout << "load texture from: " << path << std::endl;
		unsigned char* data = stbi_load(path, &m_width, &m_height, &nrChannels, 0);
		if (data) {
			cudaChannelFormatDesc channelDesc =
				cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
			cudaMallocArray(&m_cuArray, &channelDesc, m_width, m_height);
			const size_t spitch = m_width * sizeof(glm::vec4);
			// Copy data located at address h_data in host memory to device memory
			cudaMemcpy2DToArray(m_cuArray, 0, 0, data, spitch, m_width * sizeof(glm::vec4), m_height, cudaMemcpyHostToDevice);

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = m_cuArray;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 0;

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


void CudaTexture::LoadTexture(const std::string& path) {
	LoadTexture(path.c_str());
}

void CudaTexture::DeloadTexture()
{
	if (m_texObj != 0) {
		cudaDestroyTextureObject(m_texObj);
		m_texObj = 0;
	}
	if (m_cuArray != nullptr) {
		cudaFreeArray(m_cuArray);
		m_cuArray = nullptr;
	}
}
