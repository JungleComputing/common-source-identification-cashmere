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
    __global__ void normalizeKernel(const int n, float* v);
}

// This kernel is required for the Cuda version, because Cufft does not normalize the result after
// inverse fft. Our FFT invokes this kernel after Cufft.
__global__ void normalizeKernel(const int n, float* v) {
    const int bi = blockIdx.x;
    const int wti = threadIdx.y;
    const int tti = threadIdx.x;

    const int nrThreadsN = min(1024, n);
    const int nrThreadsNrThreadsN = min(32, nrThreadsN);
    const int ti = wti * (1 * nrThreadsNrThreadsN) + tti;
    if (ti < nrThreadsN) {
        const int i = bi * (1 * nrThreadsN) + ti;
        if (i < n) {
            v[2*i] = v[2*i] / n;
            v[2*i+1] = v[2*i+1] / n;
        }
    }
}
