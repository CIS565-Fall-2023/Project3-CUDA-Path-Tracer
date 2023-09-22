#include <cstdio>
#include "cpu.h"
#include "common.h"
#include <vector>

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int left = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] > 0) {
                    odata[left] = idata[i];
                    ++left;
                }
            }
            timer().endCpuTimer();
            return left;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* zo = new int[n];
            for (int i = 0; i < n; i++) {
                zo[i] = (idata[i] > 0);
            }
            int sum = 0;
            int* s = new int[n];
            for (int i = 0; i < n; i++) {
                s[i] = sum;
                sum += zo[i];
            }
            int count = s[n - 1] + (zo[n - 1] > 0);
            for (int i = 0; i < n; i++) {
                if (zo[i] > 0) {
                    odata[s[i]] = idata[i];
                }
                
            }
            delete[] zo;
            delete[] s;
            timer().endCpuTimer();
            return count;
        }

        void sort(int n, int* odata, int* idata) {
            timer().startCpuTimer();
            // std::vector<int> vec();
            std::sort(idata, idata + n);
            memcpy(odata, idata, sizeof(int) * n);
            
            timer().endCpuTimer();
        }
    }
}
