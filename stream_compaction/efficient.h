#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanSharedHelper(int n, int depth, int* odata, int* idata);
        void scanShared(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        // helper
        void createSumArr(int depth, int next_power_of_two);

        void freeSumArr(int depth);
    }
}
