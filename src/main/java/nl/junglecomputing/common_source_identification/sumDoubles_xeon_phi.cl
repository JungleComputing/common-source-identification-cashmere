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



#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void sumDoublesKernel(const int n, __global double* result, const __global double* input) {
    const int ti = get_group_id(0);
    const int vi = get_local_id(0);

    const int nrThreads = 1;
    const int nrVectors = 16;
    if (vi == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += input[i];
        }
        *result = sum;
    }
}
