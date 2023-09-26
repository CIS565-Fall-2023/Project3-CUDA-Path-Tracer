#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define ELEMENTS_PER_BLOCK 128
#define TWICE_ELEMENTS_PER_BLOCK 256

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

// sum aux arrays
int** sumArr;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int* x) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}

            int k = index * offset;
            x[k + offset - 1] += x[k + (offset >> 1) - 1];
        }

        __global__ void kernDownSweep(int n, int offset, int* x) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int left = (1 + (index << 1)) * offset - 1,
                right = left + offset;

            int t = x[left];
            x[left] = x[right];
            x[right] += t;
        }

        __global__ void kernScanShared(int n, int* sums, int* odata, int* idata) {
            extern __shared__ int temp[];

            int curBlockSize = n < TWICE_ELEMENTS_PER_BLOCK ? n : TWICE_ELEMENTS_PER_BLOCK;

            int tid = threadIdx.x,
                index = tid + blockIdx.x * TWICE_ELEMENTS_PER_BLOCK;

            int offset = 1, ai = tid, bi = tid + ELEMENTS_PER_BLOCK,
                bankOffsetA = CONFLICT_FREE_OFFSET(ai), bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + bankOffsetA] = index < n ? idata[index] : 0;
            temp[bi + bankOffsetB] = (index < n - ELEMENTS_PER_BLOCK) ? idata[index + ELEMENTS_PER_BLOCK] : 0;

            // up sweep
            for (int d = curBlockSize >> 1; d > 0; d >>= 1) {
                __syncthreads();

                if (tid < d) {
                    int ai = ((tid << 1) + 1) * offset - 1;
                    int bi = ai + offset;

                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    temp[bi] += temp[ai];
                }

                offset <<= 1;
            }

            if (tid == 0) {
                sums[blockIdx.x] = temp[curBlockSize - 1 + CONFLICT_FREE_OFFSET(curBlockSize - 1)];
                temp[curBlockSize - 1 + CONFLICT_FREE_OFFSET(curBlockSize - 1)] = 0;
            }

            // down sweep
           for (int d = 1; d < curBlockSize; d <<= 1) {
                offset >>= 1;
                __syncthreads();

                if (tid < d) {
                    int ai = ((tid << 1) + 1) * offset - 1;
                    int bi = ai + offset;

                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }

           __syncthreads();

            odata[index] = index < n ? temp[ai + bankOffsetA] : 0;
            odata[index + ELEMENTS_PER_BLOCK] = (index < n - ELEMENTS_PER_BLOCK) ? temp[bi + bankOffsetB] : 0;
		}

        __global__ void kernAdd(int n, int* odata, const int* idata) {
            int index = blockIdx.x * TWICE_ELEMENTS_PER_BLOCK + threadIdx.x;
            if (index >= n) {
                return;
            }

            odata[index] += idata[blockIdx.x];
            if (index < n - ELEMENTS_PER_BLOCK) {
                odata[index + ELEMENTS_PER_BLOCK] += idata[blockIdx.x];
            }
        }

        void createSumArr(int depth, int next_power_of_two) {
            sumArr = (int**)malloc(sizeof(int*) * depth);
            int blockCnt = (int)ceil((float)next_power_of_two / (float)TWICE_ELEMENTS_PER_BLOCK);
            for (int i = 0; i < depth; ++i) {
                cudaMalloc((void**)&(sumArr[i]), blockCnt * sizeof(int));
                blockCnt = (int)ceil((float)blockCnt / (float)TWICE_ELEMENTS_PER_BLOCK);
            }
        }

        void freeSumArr(int depth) {
            for (int i = 0; i < depth; ++i) {
                cudaFree(sumArr[i]);
            }
            free(sumArr);
        }

        void scanSharedHelper(int n, int depth, int* odata, int* idata) {
            int blockCnt = (int)ceil((float)n / (float)TWICE_ELEMENTS_PER_BLOCK);

            kernScanShared << <blockCnt, ELEMENTS_PER_BLOCK, TWICE_ELEMENTS_PER_BLOCK * sizeof(int) >> > (n, sumArr[depth], odata, idata);
            if (blockCnt > 1) {
                scanSharedHelper(blockCnt, depth + 1, sumArr[depth], sumArr[depth]);
				kernAdd << <blockCnt, ELEMENTS_PER_BLOCK >> > (n, odata, sumArr[depth]);
            }
        }

        void scanShared(int n, int* odata, const int* idata) {
            int max_d = ilog2ceil(n);
            int next_power_of_two = 1 << max_d;

            int* in, int* out;
            cudaMalloc((void**)&in, next_power_of_two * sizeof(int));
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            createSumArr(max_d, next_power_of_two);

            timer().startGpuTimer();

            scanSharedHelper(next_power_of_two, 0, out, in);

            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(in);
            cudaFree(out);

            freeSumArr(max_d);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // memory operation
            int max_d = ilog2ceil(n);
            int next_power_of_two = 1 << max_d;

            int* x;
            cudaMalloc((void**)&x, next_power_of_two * sizeof(int));
            cudaMemcpy(x, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // optimize block size
            // block size should be multiple of warp size and less than max threads per block
            // grid size should be multiple of number of SMs
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            int minBlockSize = prop.warpSize, maxBlockSize = prop.maxThreadsPerBlock, sms = prop.multiProcessorCount;

            timer().startGpuTimer();

            // TODO
            int step = 1, threadCount = next_power_of_two;

            // up-sweep
            for (int d = 0; d < max_d; ++d) {
                step <<= 1;
                threadCount >>= 1;

                int curBlockSize = std::max(minBlockSize, std::min(threadCount, maxBlockSize));
                int curGridSize = (threadCount + curBlockSize - 1) / curBlockSize;
                curGridSize = std::ceil(curGridSize / (float)sms) * sms;

				kernUpSweep<<<curGridSize, curBlockSize >>>(threadCount, step, x);
			}

            // down-sweep
            cudaMemset(x + next_power_of_two - 1, 0, sizeof(int));

            for (int d = max_d - 1; d >= 0; --d) {
                step >>= 1;
                
                int smMultiple = 1 << ((int)std::ceil(threadCount / (float)sms));
                int curBlockSize = std::max(minBlockSize, std::min(threadCount, std::min(smMultiple, maxBlockSize)));
                int curGridSize = (threadCount + curBlockSize - 1) / curBlockSize;
                curGridSize = std::ceil(curGridSize / (float)sms) * sms;

                kernDownSweep<<<curGridSize, curBlockSize >>>(threadCount, step, x);
                threadCount <<= 1;
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, x, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(x);
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
        int compact(int n, int *odata, const int *idata) {
            int* bools, *scanArr, *out, *in;
            cudaMalloc((void**)&bools, n * sizeof(int));
            cudaMalloc((void**)&scanArr, n * sizeof(int));
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMalloc((void**)&in, n * sizeof(int));

            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            //timer().startGpuTimer();
            
            // TODO
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            int minBlockSize = prop.warpSize, maxBlockSize = prop.maxThreadsPerBlock, sms = prop.multiProcessorCount;
            int curBlockSize = std::max(minBlockSize, std::min(n, maxBlockSize));
            int gridSize = (n + curBlockSize - 1) / curBlockSize;

            // Step 1: Compute temporary array of 0s and 1s
            StreamCompaction::Common::kernMapToBoolean<<<gridSize, curBlockSize >>>(n, bools, in);

            // Step2: Run exclusive scan on tempArr
            scan(n, scanArr, bools);

            // Step 3: Scatter
            StreamCompaction::Common::kernScatter<<<gridSize, curBlockSize >>>(n, out, in, bools, scanArr);

            //timer().endGpuTimer();

            int count = 0, lastScan = 0;
            cudaMemcpy(&count, scanArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastScan, bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(scanArr);
            cudaFree(out);
            cudaFree(in);

            return count + lastScan;
        }
    }
}
