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
    __global__ void sumDoublesKernel(const int n, double* result, const double* input);
}

__global__ void sumDoublesKernel(const int n, double* result, const double* input) {
    const int wti = threadIdx.y;
    const int tti = threadIdx.x;

    const int nrThreads = 256;
    const int nrThreadsNrThreads = min(32, nrThreads);
    __shared__ double reduceMem[256];
    const int ti = wti * (1 * nrThreadsNrThreads) + tti;
    if (ti < nrThreads) {
        if (ti < n) {
            double sum = 0.0;
            for (int i = ti; i < n; i += nrThreads) {
                sum += input[i];
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
                *result = reduceMem[0];
            }
        }
    }
}
