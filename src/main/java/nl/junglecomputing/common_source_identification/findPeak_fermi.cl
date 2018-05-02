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







__kernel void findPeakKernel(const int nrBlocks, const int n, __global float* peak, __global float* peaks, __global int* indicesPeak, const __global float* input) {
    const int bi = get_group_id(0);
    const int wti = get_local_id(1);
    const int tti = get_local_id(0);

    const int nrThreads = 256;
    const int stepSize = nrBlocks * nrThreads;
    const int nrThreadsNrThreads = min(32, nrThreads);
    const int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ?
        1 :
        nrThreads % (1 * nrThreadsNrThreads) == 0 ?
            nrThreads / (1 * nrThreadsNrThreads) :
            nrThreads / (1 * nrThreadsNrThreads) + 1
    ;
    __local float reduceMem[256];
    __local int indexMem[256];
    const int ti = wti * (1 * nrThreadsNrThreads) + tti;
    if (ti < nrThreads) {
        float max = -1.0;
        int index = -1;
        for (int i = bi * nrThreads + ti; i < n; i += stepSize) {
            const float val = fabs(input[i * 2 + 0]);
            if (val > max) {
                max = val;
                index = i;
            }
        }
        reduceMem[ti] = max;
        indexMem[ti] = index;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = nrThreads / 2; i > 0; i >>= 1) {
            if (ti < i) {
                const float v1 = reduceMem[ti];
                const float v2 = reduceMem[ti + i];
                if (v2 > v1) {
                    reduceMem[ti] = v2;
                    indexMem[ti] = indexMem[ti + i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (ti == 0) {
            peaks[bi] = reduceMem[0];
            indicesPeak[bi] = indexMem[0];
            if (bi == 0) {
	      *peak = input[(n - 1) * 2 + 0];
            }
        }
    }
}
