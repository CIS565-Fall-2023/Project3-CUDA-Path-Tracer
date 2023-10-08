#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <ostream>

template<typename T>
class CudaMemory {
private:
	T* dev_ptr;
	bool m_malloc;
	int m_size;
	CudaMemory(const CudaMemory& _c);
	CudaMemory& operator=(const CudaMemory& _c);
	void free() {
		if (m_malloc)cudaFree(dev_ptr);
	}
public:
	CudaMemory()
		:dev_ptr(nullptr),m_malloc(false),m_size(0)
	{}
	~CudaMemory() {
		free();
	}
	void malloc(int size, const char* error_msg = "malloc error") {
		free();
		cudaMalloc((void**)&dev_ptr, size * sizeof(T));

		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			fprintf(stderr, "Cuda error: %s: %s.\n", error_msg, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		m_malloc = true;
		m_size = size;
	}
	T* get() {
		return dev_ptr;
	}
	int size() {
		return m_size;
	}
};