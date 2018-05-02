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




__kernel void transposeKernel(const int h, const int w, __global float* output, const __global float* input) {
    const int i = get_group_id(1);
    const int tj = get_group_id(0);
    const int vj = get_local_id(0);

    const int nrVectorsW = min(16, w);
    const int nrThreadsW = w == 1 * nrVectorsW ?
        1 :
        w % (1 * nrVectorsW) == 0 ?
            w / (1 * nrVectorsW) :
            w / (1 * nrVectorsW) + 1
    ;
    const int j = tj * (1 * nrVectorsW) + vj;
    if (j < w) {
        output[j + i * (1 * w)] = input[i + j * (1 * h)];
    }
}
