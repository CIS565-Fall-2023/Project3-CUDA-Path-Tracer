#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                bools[index] = (idata[index] > 0);
            }
        }

        /**
         * Maps an array to an array according to ith bit value for stream compaction.
         */
        __global__ void kernMapBitToBoolean(int n, int i, int *bools, int* ebools, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                bools[index] = ((idata[index] >> i) & 1);
                ebools[index] = !((idata[index] >> i) & 1);
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n && bools[index] > 0) {
                odata[indices[index]] = idata[index];
            }
        }

        __global__ void kernFixedScatter(int n, int* odata,
            const int* idata,  const int* indices) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                odata[indices[index]] = idata[index];
            }
        }

        //__global__ void kernReverseBoolean(int n, int* ebools, const int* bools)
        //{
        //    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //    if (index < n) {
        //        ebools[index] = (bools[index] == 0);
        //    }
        //}

        __global__ void kernOrderCheck(int n, const int* data, int* sign)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n - 1 && data[index] > data[index + 1]) {
                sign[0] = -1;
            }
        }
    }
}
