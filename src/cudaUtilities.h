#pragma once

#include <vector>
#include<cuda_runtime.h>

template<typename T>
void MallocArrayOnCuda(T*& dev_array, std::vector<T>& host_array)
{
	if (host_array.size() > 0 && !dev_array)
	{
		cudaMalloc(&dev_array, host_array.size() * sizeof(T));
		cudaMemcpy(dev_array, host_array.data(), host_array.size() * sizeof(T), cudaMemcpyHostToDevice);
	}
}