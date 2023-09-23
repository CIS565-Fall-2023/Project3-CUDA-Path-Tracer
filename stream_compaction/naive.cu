#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + blockDim.x * blockIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= (1 << (d - 1))) {
                odata[index] = idata[index - (1 << (d - 1))] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_in;
            int* dev_out;

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // allocate memory
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_in failed!");
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_out failed!");

            // populate dev_in
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy to dev_in failed!");

            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                kernNaiveScan << <gridSize, BLOCK_SIZE >> > (n, d, dev_out, dev_in);
                checkCUDAErrorFn("kernNaiveScan failed!");

                std::swap(dev_in, dev_out);
            }
            timer().endGpuTimer();

            // shift to exclusive
            int zero = 0;
            cudaMemcpy(&dev_out[0], &zero, 1 * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&dev_out[1], dev_in, (n - 1) * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAErrorFn("shift failed!");

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy to odata failed!");

            // free cuda memory
            cudaFree(dev_in);
            cudaFree(dev_out);

            cudaDeviceSynchronize();
        }
    }
}
