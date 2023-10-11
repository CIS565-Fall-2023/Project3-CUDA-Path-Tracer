#include <cuda.h>
#include <cuda_runtime.h> 
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 128
#define DOUBLE_BLOCK_SIZE 2 * BLOCK_SIZE
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


#define TIME_COMPACT 1

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
 
        __global__ void kernUpSweep(int n, int d, int* odata) {
            size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * (2 << d);
 
            if (index >= n) return;

            odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* odata) {
            size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * (2 << d);

            if (index >= n) return;

            // preserve the left child value
            int temp = odata[index + (1 << d) - 1];
            // left child copies the parent value
            odata[index + (1 << d) - 1] = odata[index + (1 << (d + 1)) - 1];
            // right child addes the parent value and the preserved left child value
            odata[index + (1 << (d + 1)) - 1] += temp;
        }

         /**
          *  apply shared memory to scan each block
          */
        __global__ void kernBlockScan(int n, int* odata, const int* idata, int* blockSums) {
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            size_t index = blockIdx.x * blockDim.x + thid;

            // Load input into shared memory with boundary checks
            temp[2 * thid] = (2 * index < n) ? idata[2 * index] : 0;
            temp[2 * thid + 1] = (2 * index + 1 < n) ? idata[2 * index + 1] : 0;
            __syncthreads();

            int offset = 1;
           
            // Up-sweep (reduce) phase
            for (int d = blockDim.x; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            // Clear last element
            if (thid == 0) {
                blockSums[blockIdx.x] = temp[2 * blockDim.x - 1];
                temp[2 * blockDim.x - 1] = 0;
            }

            // Down-sweep phase
            for (int d = 1; d < 2 * blockDim.x; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            // Write results to device memory with boundary checks
            if (2 * index < n) {
                odata[2 * index] = temp[2 * thid];
            }

            if (2 * index + 1 < n) {
                odata[2 * index + 1] = temp[2 * thid + 1];
            }
        }

        __global__ void kernAddScannedBlockSums(int n, int* odata, const int* blockSums) {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int blockSum = (blockIdx.x > 0) ? blockSums[blockIdx.x - 1] : 0;

            if (2 * index < n) {
                odata[2 * index] += blockSum;
            }

            if (2 * index + 1 < n) {
                odata[2 * index + 1] += blockSum;
            }
        }

        /**
         * Performs prefix-sum (aks scan) on idata using the shared memory, storing the result into odata
         */
        void scanShared(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out, * dev_blockSums;
            
            const int log2ceil = ilog2ceil(n);
            const int fullSize = 1 << log2ceil;

            int gridSize = (fullSize + DOUBLE_BLOCK_SIZE - 1) / (DOUBLE_BLOCK_SIZE);
            // printf("gridSize: %d\n", gridSize);

            // allocate gpu memory
            cudaMalloc((void**)&dev_in, fullSize * sizeof(int));
            cudaMemset(dev_in, 0, fullSize * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_out, n * sizeof(int));

            cudaMalloc((void**)&dev_blockSums, gridSize * sizeof(int));
            checkCUDAErrorFn("malloc dev_blockSums failed!");

            timer().startGpuTimer();
            kernBlockScan << <gridSize, BLOCK_SIZE, DOUBLE_BLOCK_SIZE * sizeof(int) >> > (fullSize, dev_out, dev_in, dev_blockSums);
            checkCUDAErrorFn("blockScan failed!");

            int* blockSums = new int[gridSize];
            cudaMemcpy(blockSums, dev_blockSums, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
   
            for (int i = 1; i < gridSize; ++i) {
                blockSums[i] += blockSums[i - 1];
            }

            cudaMemcpy(dev_blockSums, blockSums, gridSize * sizeof(int), cudaMemcpyHostToDevice);
            delete[] blockSums;

            kernAddScannedBlockSums << <gridSize, BLOCK_SIZE >> > (fullSize, dev_out, dev_blockSums);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_blockSums);
        }

        void scanHelper(int n, int log2ceil, int* dev_out) {
            // up sweep 
            for (int d = 0; d <= log2ceil - 1; ++d) {
                // Adjust the grid size based on the depth of the sweep
                int gridSize = (n / (2 << d) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweep << <gridSize, BLOCK_SIZE >> > (n, d, dev_out);
                checkCUDAErrorFn("up sweep failed!");
            }

            // set the last value to 0
            cudaMemset(dev_out + n - 1, 0, sizeof(int));
            checkCUDAErrorWithLine("set the last value to zero failed!");

            // down sweep
            for (int d = log2ceil - 1; d >= 0; --d) {
                // Adjust the grid size based on the depth of the sweep
                int gridSize = (n / (2 << d) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <gridSize, BLOCK_SIZE >> > (n, d, dev_out);
                checkCUDAErrorFn("down sweep failed");
            }

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_out;

            const int log2ceil = ilog2ceil(n);
            const long int fullSize = 1 << log2ceil;

            cudaMalloc((void**)&dev_out, fullSize * sizeof(int));
            cudaMemset(dev_out, 0, fullSize * sizeof(int));
            cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanHelper(fullSize, log2ceil, dev_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_out);
        }


        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out, * dev_bools, * dev_scan;

            int boolLastVal, scanLastVal;

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_in failed!");
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy idata to dev_in failed!");

            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_out failed!");

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_bools failed!");

#if TIME_COMPACT
            const int log2ceil = ilog2ceil(n);
            const long int fullSize = 1 << log2ceil;

            cudaMalloc((void**)&dev_scan, fullSize * sizeof(int));
            checkCUDAErrorFn("malloc dev_scan failed!");
            cudaMemset(dev_scan, 0, n * sizeof(int));
#else
            cudaMalloc((void**)&dev_scan, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_scan failed!");
#endif

#if TIME_COMPACT
            timer().startGpuTimer();
#endif
            // map the bool array
            StreamCompaction::Common::kernMapToBoolean << <gridSize, BLOCK_SIZE >> > (n, dev_bools, dev_in);
            checkCUDAErrorFn("map bool array failed!");

            
#if TIME_COMPACT
            // scan the bool array
            cudaMemcpy(dev_scan, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // up sweep
            for (int d = 0; d <= log2ceil - 1; ++d) {
                int dynamicGridSize = (fullSize / (2 << d) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweep << <dynamicGridSize, BLOCK_SIZE >> > (fullSize, d, dev_scan);
                checkCUDAErrorFn("up sweep failed!");
            }

            // set the last value to 0
            cudaMemset(dev_scan + fullSize - 1, 0, sizeof(int));
            
            // down sweep
            for (int d = log2ceil - 1; d >= 0; --d) {
                int dynamicGridSize = (fullSize / (2 << d) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <dynamicGridSize, BLOCK_SIZE >> > (fullSize, d, dev_scan);
                checkCUDAErrorFn("down sweep failed");
            }
#else 
            // scan(n, dev_scan, dev_bools);
            scanShared(n, dev_scan, dev_bools);
#endif
            // scatter
            StreamCompaction::Common::kernScatter << <gridSize, BLOCK_SIZE >> > (n, dev_out, dev_in, dev_bools, dev_scan);
            checkCUDAErrorFn("scatter failed!");
#if TIME_COMPACT
            timer().endGpuTimer();
#endif
            // store the last value of the bool array
            cudaMemcpy(&boolLastVal, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy last bool value to host failed!");

            // store the last value of the scan results
            cudaMemcpy(&scanLastVal, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy last bool value to host failed!");

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_out to odata failed!");

            // free memory
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bools);
            cudaFree(dev_scan);

            return scanLastVal + boolLastVal;
        }
    }
}
