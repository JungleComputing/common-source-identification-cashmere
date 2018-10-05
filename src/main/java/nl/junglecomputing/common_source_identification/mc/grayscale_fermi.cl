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

typedef struct __attribute__ ((packed)) {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;



__kernel void grayscaleKernel(const int n, __global float* output, const __global Color* input) {
    const int bi = get_group_id(0);
    const int wti = get_local_id(1);
    const int tti = get_local_id(0);

    const int nrThreadsN = min(1024, n);
    const int nrBlocksN = n == 1 * nrThreadsN ?
        1 :
        n % (1 * nrThreadsN) == 0 ?
            n / (1 * nrThreadsN) :
            n / (1 * nrThreadsN) + 1
    ;
    const int nrThreadsNrThreadsN = min(32, nrThreadsN);
    const int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ?
        1 :
        nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ?
            nrThreadsN / (1 * nrThreadsNrThreadsN) :
            nrThreadsN / (1 * nrThreadsNrThreadsN) + 1
    ;
    const int ti = wti * (1 * nrThreadsNrThreadsN) + tti;
    if (ti < nrThreadsN) {
        const int i = bi * (1 * nrThreadsN) + ti;
        if (i < n) {
            const float r = (float) input[i].r;
            const float g = (float) input[i].g;
            const float b = (float) input[i].b;
            output[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
}
