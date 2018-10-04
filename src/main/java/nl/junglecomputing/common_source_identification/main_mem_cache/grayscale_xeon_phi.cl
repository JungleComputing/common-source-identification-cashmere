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
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;



__kernel void grayscaleKernel(const int n, __global float* output, const __global Color* input) {
    const int ti = get_group_id(0);
    const int vi = get_local_id(0);

    const int nrVectorsN = min(16, n);
    const int nrThreadsN = n == 1 * nrVectorsN ?
        1 :
        n % (1 * nrVectorsN) == 0 ?
            n / (1 * nrVectorsN) :
            n / (1 * nrVectorsN) + 1
    ;
    const int i = ti * (1 * nrVectorsN) + vi;
    if (i < n) {
        const float r = (float) input[i].r;
        const float g = (float) input[i].g;
        const float b = (float) input[i].b;
        output[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
}
