#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanLine(int N, int offsetBase, int* odata, const int * idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1 << offsetBase;
            if (index < offset) {
                odata[index] = idata[index];
            }
            else if (index < N) {
                odata[index] = idata[index] + idata[index - offset];
            }
        }

        __global__ void shiftToExclusive(int N, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < N) {
                odata[index + 1] = idata[index];
            }
        }

        const int blockSize = 64;

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_idata failed!");

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (int i = 1; i <= ilog2ceil(n); i++) {
                scanLine << <fullBlocksPerGrid, blockSize >> > (n, i - 1, dev_odata, dev_idata);
                int* temp = dev_odata;
                dev_odata = dev_idata;
                dev_idata = temp;
            }

            dim3 blockNum((n - 1 + blockSize - 1) / blockSize);
            shiftToExclusive << <blockNum, blockSize >> > (n - 1, dev_odata, dev_idata);
            
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy idata -> dev_idata failed!");
            odata[0] = 0;

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree failed!");
        }
    }
}
