#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanHelper(int n, int log2ceil, int* dev_out);

        void scan(int n, int *odata, const int *idata);

        void scanShared(int n, int* odata, const int* idata);

        int compact(int n, int *odata, const int *idata);
    }
}
