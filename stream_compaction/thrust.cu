#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_idata failed!");

            timer().startGpuTimer();

            thrust::device_ptr<int> dv_in(dev_idata);
            thrust::device_ptr<int> dv_out(dev_odata);
            thrust::exclusive_scan(dv_in, dv_in + n, dv_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata -> odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        void sort(int n, int* odata, const int* idata) {
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_idata failed!");

            timer().startGpuTimer();

            thrust::device_ptr<int> dv_in(dev_idata);
            thrust::sort(thrust::device, dev_idata, dev_idata + n);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata -> odata failed!");
            cudaFree(dev_idata);
        }
    }
}
