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




__kernel void varianceZeroMeanKernel(const int n, __global float* variance, const __global float* input) {
    const int t = get_group_id(0);
    const int v = get_local_id(0);

    const int nrThreads = 1;
    const int nrVectors = 16;
    if (v == 0) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += input[i] * input[i];
        }
        *variance = sum * n / (n - 1);
    }
}