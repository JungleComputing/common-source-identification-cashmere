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



extern "C" {
    __global__ void maxLocFloatsKernel(const int n, float* peak, int* indexPeak, const float* peaks, const int* indicesPeak);
}




__global__ void maxLocFloatsKernel(const int n, float* peak, int* indexPeak, const float* peaks, const int* indicesPeak) {
    const int wti = threadIdx.y;
    const int tti = threadIdx.x;

    const int nrThreads = 256;
    const int nrThreadsNrThreads = min(32, nrThreads);
    __shared__ float reduceMem[256];
    __shared__ int indexMem[256];
    const int ti = wti * (1 * nrThreadsNrThreads) + tti;
    if (ti < nrThreads) {
        float max = -1.0;
        int index = -1;
        for (int i = ti; i < n; i += nrThreads) {
            const float val = fabs(peaks[i]);
            if (val > max) {
                max = val;
                index = indicesPeak[i];
            }
        }
        reduceMem[ti] = max;
        indexMem[ti] = index;
        __syncthreads();
        for (int i = nrThreads / 2; i > 0; i >>= 1) {
            if (ti < i) {
                const float v1 = reduceMem[ti];
                const float v2 = reduceMem[ti + i];
                if (v2 > v1) {
                    reduceMem[ti] = v2;
                    indexMem[ti] = indexMem[ti + i];
                }
            }
            __syncthreads();
        }
        if (ti == 0) {
            *peak = reduceMem[0];
            *indexPeak = indexMem[0];
        }
    }
}
