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



__kernel void computeSquaredMagnitudesKernel(const int h, const int w, __global float* output, const __global float* input) {
    const int i = get_group_id(1);
    const int bj = get_group_id(0);
    const int wtj = get_local_id(1);
    const int ttj = get_local_id(0);

    const int nrThreadsW = min(1024, w);
    const int nrBlocksW = w == 1 * nrThreadsW ?
        1 :
        w % (1 * nrThreadsW) == 0 ?
            w / (1 * nrThreadsW) :
            w / (1 * nrThreadsW) + 1
    ;
    const int nrThreadsNrThreadsW = min(32, nrThreadsW);
    const int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ?
        1 :
        nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ?
            nrThreadsW / (1 * nrThreadsNrThreadsW) :
            nrThreadsW / (1 * nrThreadsNrThreadsW) + 1
    ;
    const int tj = wtj * (1 * nrThreadsNrThreadsW) + ttj;
    if (tj < nrThreadsW) {
        const int j = bj * (1 * nrThreadsW) + tj;
        if (j < w) {
	  const float real = input[(j + i * (1 * w)) * 2 + 0];
	  const float imag = input[(j + i * (1 * w)) * 2 + 1];
            output[j + i * (1 * w)] = real * real + imag * imag;
        }
    }
}
