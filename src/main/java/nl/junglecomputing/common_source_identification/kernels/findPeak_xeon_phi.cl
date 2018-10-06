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







__kernel void findPeakKernel(const int nrBlocks, const int n, __global float* peak, __global float* peaks, __global int* indicesPeak, const __global Complex* input) {
    const int ti = get_group_id(0);
    const int vi = get_local_id(0);

    const int nrThreads = nrBlocks;
    const int nrVectors = 16;
    const int nrEls = n / nrThreads;
    if (vi == 0) {
        float max = -1.0;
        int index = -1;
        int start = ti * nrEls;
        int end = start + nrEls;
        for (int i = start; i < end; i++) {
            const float val = fabs(input[i].real);
            if (val > max) {
                max = val;
                index = i;
            }
        }
        if (ti == nrThreads - 1) {
            *peak = input[n - 1].real;
            start = nrThreads * nrEls;
            end = min(start + nrEls, n);
            for (int i = start; i < end; i++) {
                const float val = fabs(input[i].real);
                if (val > max) {
                    max = val;
                    index = i;
                }
            }
        }
        peaks[ti] = max;
        indicesPeak[ti] = index;
    }
}
