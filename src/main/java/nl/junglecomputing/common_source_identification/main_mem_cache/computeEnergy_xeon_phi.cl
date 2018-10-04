// xeon_phi

/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef struct __attribute__ ((packed)) {
    float real;
    float imag;
} Complex;
#define SQUARE_SIZE (11)
#define RADIUS (SQUARE_SIZE / 2)

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void computeEnergyKernel(const int nrBlocks, const int h, const int w, __global double* energy, const __global int* indexPeak, const __global Complex* input) {
    const int ti = get_group_id(0);
    const int vi = get_local_id(0);

    const int nrThreads = nrBlocks;
    const int nrVectors = 16;
    const int n = h * w;
    const int nrEls = n / nrThreads + 1;
    double sum = 0.0;
    if (vi == 0) {
        const int indexPeakY = indexPeak[0] / w;
        const int indexPeakX = indexPeak[0] - indexPeakY * w;
        const int start = ti * nrEls;
        const int end = min(start + nrEls, n);
        for (int i = start; i < end; i++) {
            const int row = i / w;
            const int col = i - row * w;
            const bool inRowPeak = row > indexPeakY - RADIUS && row < indexPeakY + RADIUS;
            const bool inColPeak = col > indexPeakX - RADIUS && col < indexPeakX + RADIUS;
            if (!(inRowPeak && inColPeak)) {
                const double val = input[col + row * (1 * w)].real;
                sum += val * val;
            }
        }
        energy[ti] = sum / ((double) n - SQUARE_SIZE * SQUARE_SIZE);
    }
}
