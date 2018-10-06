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




__kernel void varianceZeroMeanKernel(const int n, __global float* variance, const __global float* input) {
    const int b = get_group_id(0);
    const int wti = get_local_id(1);
    const int tti = get_local_id(0);

    const int nrBlocks = 1;
    const int nrThreads = 1024;
    const int nrThreadsNrThreads = min(32, nrThreads);
    const int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ?
        1 :
        nrThreads % (1 * nrThreadsNrThreads) == 0 ?
            nrThreads / (1 * nrThreadsNrThreads) :
            nrThreads / (1 * nrThreadsNrThreads) + 1
    ;
    __local float reduceMem[1024];
    const int ti = wti * (1 * nrThreadsNrThreads) + tti;
    if (ti < nrThreads) {
        if (ti < n) {
            float sum = 0.0;
            for (int i = ti; i < n; i += nrThreads) {
                sum += input[i] * input[i];
            }
            reduceMem[ti] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = nrThreads / 2; i > 0; i >>= 1) {
                if (ti < i) {
                    reduceMem[ti] += reduceMem[ti + i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (ti == 0) {
                *variance = reduceMem[0] * n / (n - 1);
            }
        }
    }
}
