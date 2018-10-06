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



__kernel void toComplexKernel(const int n, __global Complex* output, const __global float* input) {
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
        output[i].real = input[i];
        output[i].imag = 0.0;
    }
}
