#include <cstdio>
#include "cpu.h"

#include "common.h"

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
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
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
            int index = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[index] = idata[i];
                    index++;
                }
            }

            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* bools = new int[n];
            int* scanResults = new int[n];

            // map the bools array 
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // run exclusive scan
            scanResults[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanResults[i] = scanResults[i - 1] + bools[i - 1];
            }

            // scatter
            for (int i = 0; i < n; ++i) {
                if (bools[i] != 0) {
                    odata[scanResults[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return scanResults[n - 1] + bools[n - 1];
        }
    }
}
