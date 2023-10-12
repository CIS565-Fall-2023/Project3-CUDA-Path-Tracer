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

        void scanImpl(int n, int* odata, const int* idata) {
			odata[0] = 0;
			for (int i = 0; i < n - 1; ++i) {
				odata[i + 1] = odata[i] + idata[i];
			}
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // TODO
            scanImpl(n, odata, idata);

            timer().endCpuTimer();
        }

        void sort(int n, int* odata, const int* idata) {
            memcpy(odata, idata, n * sizeof(int));

            timer().startCpuTimer();

			std::sort(odata, odata + n);

			timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // TODO
            int index = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
					odata[index++] = idata[i];
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
            // TODO
            int* scanArr = new int[n];
            int* tempArr = new int[n];

            timer().startCpuTimer();

            // Step 1: Compute temporary array of 0s and 1s
            for (int i = 0; i < n; ++i) {
                tempArr[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Step2: Run exclusive scan on tempArr
            scanImpl(n, scanArr, tempArr);

            // Step 3: Scatter
            for (int i = 0; i < n; ++i) {
                if (tempArr[i] == 1) {
                    odata[scanArr[i]] = idata[i];
                }
            }

            int count = scanArr[n - 1] + tempArr[n - 1];

            delete[] tempArr;
            delete[] scanArr;

            timer().endCpuTimer();
            return count;
        }
    }
}
