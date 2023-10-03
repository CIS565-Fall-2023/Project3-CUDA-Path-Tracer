#include "cudaTexture.h"

#include "stb_image.h"
#include <iostream>

Texture2D::Texture2D()
	:m_RawImg(nullptr), 
    m_Width(0), 
    m_Height(0),
    m_TexObj(0),
    m_TexArray(nullptr)
{
}

Texture2D::Texture2D(const std::string& img_path, bool flip_v)
    :Texture2D()
{
    Create(img_path, flip_v);
}

Texture2D::~Texture2D()
{
	Free();
}

void Texture2D::Create(const std::string& img_path, bool flip_v)
{
	Free();

	// read image
	stbi_set_flip_vertically_on_load(flip_v);
	m_RawImg = stbi_loadf(img_path.c_str(), &m_Width, &m_Height, nullptr, 4);
	assert(m_RawImg != nullptr);

	// create array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaMallocArray(&m_TexArray, &channelDesc, m_Width, m_Height);

    int spitch = m_Width * sizeof(float4);
    const int& width = spitch;
    // send image to device
    cudaMemcpy2DToArray(m_TexArray, 0, 0, m_RawImg, 
                            spitch,
                            width,
                            m_Height, cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_TexArray;

    // texture description
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&m_TexObj, &resDesc, &texDesc, nullptr);
}

void Texture2D::Free()
{
    std::cout << "free texture" << std::endl;
	if (!m_RawImg) stbi_image_free(m_RawImg);
	m_RawImg = nullptr;

	m_Width = m_Height = 0;

    if (m_TexObj != 0)
    {
        cudaDestroyTextureObject(m_TexObj);
        m_TexObj = 0;
    }
    SafeCudaFreeArray(m_TexArray);
}

CudaTexture2D::CudaTexture2D(const cudaTextureObject_t& tex_obj)
	:m_TexObj(tex_obj)
{
}