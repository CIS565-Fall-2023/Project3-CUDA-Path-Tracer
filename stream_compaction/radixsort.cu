#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMaxElement(int n, int* res, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (*res < idata[index]) {
                *res = idata[index];
            }
            
        }

        __global__ void kernMapToBoolean(int n, int bit, int* e, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}

			e[index] = (idata[index] & (1 << bit)) == 0;
        }

        __global__ void kernScatter(int n, int *cur, int *last, int *e, int *f) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int totalFalses = f[n - 1] + e[n - 1];

            int t = index - f[index] + totalFalses;
            int d = e[index] ? f[index] : t;
            last[d] = cur[index];
        }


        void sort(int n, int* odata, const int* idata) {
            int *last, *cur, *out;
            cudaMalloc((void**)&last, n * sizeof(int)); // last array i
            cudaMalloc((void**)&cur, n * sizeof(int)); // current array i
            cudaMalloc((void**)&out, n * sizeof(int));

            cudaMemcpy(cur, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int *e, *f;
            cudaMalloc((void**)&e, n * sizeof(int));
            cudaMalloc((void**)&f, n * sizeof(int));

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            int minBlockSize = prop.warpSize, maxBlockSize = prop.maxThreadsPerBlock, sms = prop.multiProcessorCount;
            int curBlockSize = std::max(minBlockSize, std::min(n, maxBlockSize));
            int curGridSize = (n + curBlockSize - 1) / curBlockSize;

            int* maxElement;
            cudaMalloc((void**)&maxElement, sizeof(int));
            kernMaxElement << <curGridSize, curBlockSize >> > (n, maxElement, cur);
            
            int maxVal;
            cudaMemcpy(&maxVal, maxElement, sizeof(int), cudaMemcpyDeviceToHost);
            int numBits = ilog2ceil(maxVal);

            int max_d = ilog2ceil(n);
            int next_power_of_two = 1 << max_d;
            StreamCompaction::Efficient::createSumArr(max_d, next_power_of_two);

            timer().startGpuTimer();
            // run split on each bit
            for (int bit = 0; bit <= numBits; ++bit) {
                // find b
                // get e array
                kernMapToBoolean<<<curGridSize, curBlockSize>>>(n, bit, e, cur); // e

                // exclusive scan e
                StreamCompaction::Efficient::scanSharedHelper(next_power_of_two, 0, f, e);

                // find scatter indices

                kernScatter << <curGridSize, curBlockSize >> > (n, cur, last, e, f);
				std::swap(last, cur);
			}

            timer().endGpuTimer();

            cudaMemcpy(odata, cur, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(last);
            cudaFree(cur);
            cudaFree(out);
            cudaFree(e);
            cudaFree(f);
            cudaFree(maxElement);

            StreamCompaction::Efficient::freeSumArr(max_d);
        }
    }
}
