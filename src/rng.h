#pragma once

#include <thrust/random.h>
#include "common.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
CPU_GPU inline unsigned int utilhash(unsigned int a) 
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

class CudaRNG
{
public:
	CPU_GPU CudaRNG(int iter, int index, int depth)
	{
		int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
		rng = thrust::default_random_engine(h);
	}

	CPU_GPU float rand(const float& min = 0.f, const float& max = 1.f)
	{
		return thrust::uniform_real_distribution<float>(min, max)(rng);
	}

protected:
	thrust::default_random_engine rng;
};