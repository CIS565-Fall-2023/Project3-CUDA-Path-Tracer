#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
#include "common.cu"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void initData(int n, int max, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < (max - n)) {
                data[n + index] = 0;
            }
        }

        __global__ void changeNum(int i, int newNum, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == 0) {
                data[i] = newNum;
            }
        }
        
        __global__ void upSweep(int N, int offsetBase, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1 << offsetBase;
            
            if ((index + 1) * offset * 2 - 1 < N) {
                int right = (index + 1) * offset * 2 - 1;
                data[right] += data[right - offset];
            }
        }

        __global__ void downSweep(int N, int offsetBase, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1 << offsetBase;
            int right = (index + 1) * offset * 2 - 1;
            if (right < N) {
                int t = data[right - offset];
                data[right - offset] = data[right];
                data[right] += t;
            }
        }

        void printArr(int n, int* odata, int* dev_odata) {
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata -> odata for printArr failed!");
            for (int i = 0; i <= n / 10; i++) {
                for (int j = 0; j < 10 && j < n - 10 * i; j++) {
                    std::cout << odata[i * 10 + j] << "  ";
                }std::cout << std::endl;
            }std::cout << std::endl << std::endl;
        }

        void upAndDownSweep(int max, int blockSize, int* dev_data)
        {
            // up sweep
            int addTimes = max / 2;
            for (int i = 0; i < ilog2ceil(max); i++) {
                int bs = std::min(blockSize, addTimes);
                dim3 fullBlocksPerGrid((addTimes + bs) / bs);
                upSweep << <fullBlocksPerGrid, bs >> > (max, i, dev_data);
                addTimes /= 2;
            }

            // down sweep
            cudaMemset(dev_data + max - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data + max - 1 to 0 failed!");

            int swapTime = 1;
            for (int i = ilog2ceil(max) - 1; i >= 0; i--) {
                int bs = std::min(blockSize, swapTime);
                dim3 fullBlocksPerGrid((swapTime + bs) / bs);
                downSweep << <fullBlocksPerGrid, bs >> > (max, i, dev_data);
                swapTime *= 2;
            }
        }

        const int maxBlockSize = 64;

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const int max = 1 << ilog2ceil(n);
            // std::cout << "n = " << n << ", max = " << max << std::endl;

            int* dev_data;
            cudaMalloc((void**)&dev_data, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed!");

            timer().startGpuTimer();

            upAndDownSweep(max, maxBlockSize, dev_data);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data -> odata failed!");

            cudaFree(dev_data);
            checkCUDAError("cudaFree failed!");
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
            
            int* dev_idata;
            int* dev_odata;
            int* dev_bool;
            int* dev_sum;
            const int max = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_idata, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bool, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_sum, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_sum failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed!");

            dim3 fullBlocksPerGrid((n + maxBlockSize - 1) / maxBlockSize);

            timer().startGpuTimer();

            Common::kernMapToBoolean << < fullBlocksPerGrid, maxBlockSize >> > (n, dev_bool, dev_idata);

            int last_bool = -1;
            cudaMemcpy(&last_bool, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bool_last -> last_bool failed!");

            cudaMemcpy(dev_sum, dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_bool -> dev_sum failed!");

            upAndDownSweep(max, maxBlockSize, dev_sum);


            Common::kernScatter << <fullBlocksPerGrid, maxBlockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_sum);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata -> odata failed!");

            int res_n = -1;
            cudaMemcpy(&res_n, dev_sum + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata_last -> res_n failed!");

            // std::cout << "res_n = " << res_n << ", last_bool = " << last_bool << std::endl;

            res_n += last_bool;

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bool);
            cudaFree(dev_sum);
            checkCUDAError("cudaFree failed!");

            return res_n;
        }

        __global__ void combineRadixIdx(int n, int negCount, const int* bools, const int* f, const int* t, int* dev_idx)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                dev_idx[index] = bools[index] > 0 ? t[index] + negCount: f[index];
            }
        }

        /**
         * Performs radix sort for idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         */
        void radixSort(int n, int* odata, const int* idata) {
            int* dev_idata;
            int* dev_odata;
            int* dev_bool;
            int* dev_e;
            // int* dev_f;
            int* dev_t;
            int* dev_idx;
            int* dev_sign;
            const int max = 1 << (ilog2ceil(n - 1));
            cudaMalloc((void**)&dev_idata, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMalloc((void**)&dev_odata, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bool, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_e, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            //cudaMalloc((void**)&dev_f, max * sizeof(int));
            //checkCUDAError("cudaMalloc dev_f failed!");
            cudaMalloc((void**)&dev_t, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");
            cudaMalloc((void**)&dev_idx, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");
            cudaMalloc((void**)&dev_sign, sizeof(int));
            checkCUDAError("cudaMalloc dev_sign failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed!");
            if (max > n) {
                cudaMemset(dev_idata + n, INT_MAX, sizeof(int) * (max - n));
            }
            
            // std::cout << "radixSort, max = " << max << ", n = " << n << std::endl;

            dim3 fullBlocksPerGrid((n + maxBlockSize - 1) / maxBlockSize);
            dim3 maxFullBlocksPerGrid((max + maxBlockSize - 1) / maxBlockSize);


            timer().startGpuTimer();


            for (int i = 0; i < 32; i++)
            {
                Common::kernMapBitToBoolean << < maxFullBlocksPerGrid, maxBlockSize >> > (max, i, dev_bool, dev_e, dev_idata);

                upAndDownSweep(max, maxBlockSize, dev_e);

                cudaMemcpy(dev_t, dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy dev_bool -> dev_t failed!");
                upAndDownSweep(max, maxBlockSize, dev_t);

                int neg_count = -1;
                cudaMemcpy(&neg_count, dev_e + max - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_f_last -> neg_count failed!");
                int last_bool = -1;
                cudaMemcpy(&last_bool, dev_bool + max - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_bool_last -> last_bool failed!");
                neg_count += (last_bool == 0);

                combineRadixIdx << < maxFullBlocksPerGrid, maxBlockSize >> > (max, neg_count, dev_bool, dev_e, dev_t, dev_idx);

                Common::kernFixedScatter << < maxFullBlocksPerGrid, maxBlockSize >> > (max, dev_odata, dev_idata, dev_idx);

                int* temp = dev_odata;
                dev_odata = dev_idata;
                dev_idata = temp;

                cudaMemset(dev_sign, 1, sizeof(int));
                checkCUDAError("cudaMemset dev_sign to 1 failed!");
                Common::kernOrderCheck << < maxFullBlocksPerGrid, maxBlockSize >> > (max, dev_idata, dev_sign);
                int sign = 0;
                cudaMemcpy(&sign, dev_sign, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_sign -> sign failed!");
                if (sign > 0) {
                    break;
                }
            }

            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata -> odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bool);
            // cudaFree(dev_f);
            cudaFree(dev_t);
            cudaFree(dev_idx);
            checkCUDAError("cudaFree failed!");
        }
    }
}
