// fermi

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

/* typedef struct __attribute__ ((packed)) { */
/*     float real; */
/*     float imag; */
/* } Complex; */
#define SQUARE_SIZE (11)
#define RADIUS (SQUARE_SIZE / 2)

//function interfaces to prevent C++ garbling the kernel names
extern "C" {
    __global__ void computeEnergyKernel(const int nrBlocks, const int h, const int w, double* energy, const int* indexPeak, const float* input);
}

__global__ void computeEnergyKernel(const int nrBlocks, const int h, const int w, double* energy, const int* indexPeak, const float* input) {
    const int bi = blockIdx.x;
    const int wti = threadIdx.y;
    const int tti = threadIdx.x;

    const int nrThreads = 256;
    const int n = h * w;
    const int stepSize = nrThreads * nrBlocks;
    const int nrThreadsNrThreads = min(32, nrThreads);
    ;
    __shared__ double reduceMem[256];
    const int ti = wti * (1 * nrThreadsNrThreads) + tti;
    if (ti < nrThreads) {
        double sum = 0.0;
        if (ti < n) {
            const int indexPeakY = indexPeak[0] / w;
            const int indexPeakX = indexPeak[0] - indexPeakY * w;
            for (int i = bi * nrThreads + ti; i < n; i += stepSize) {
                const int row = i / w;
                const int col = i - row * w;
                const bool inRowPeak = row > indexPeakY - RADIUS && row < indexPeakY + RADIUS;
                const bool inColPeak = col > indexPeakX - RADIUS && col < indexPeakX + RADIUS;
                if (!(inRowPeak && inColPeak)) {
		  const double val = input[(col + row * (1 * w)) * 2 + 0];
                    sum += val * val;
                }
            }
        }
        reduceMem[ti] = sum;
        __syncthreads();
        for (int i = nrThreads / 2; i > 0; i >>= 1) {
            if (ti < i) {
                reduceMem[ti] += reduceMem[ti + i];
            }
           __syncthreads();
        }
        if (ti == 0) {
            energy[bi] = reduceMem[ti] / ((double) (n - SQUARE_SIZE * SQUARE_SIZE));
        }
    }
}
