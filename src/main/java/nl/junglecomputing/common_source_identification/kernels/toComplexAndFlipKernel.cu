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
    __global__ void toComplexAndFlipKernel(const int h, const int w, float* output, const float* input);
}


__global__ void toComplexAndFlipKernel(const int h, const int w, float* output, const float* input) {
    const int i = blockIdx.y;
    const int bj = blockIdx.x;
    const int wtj = threadIdx.y;
    const int ttj = threadIdx.x;

    const int nrThreadsW = min(1024, w);
    const int nrThreadsNrThreadsW = min(32, nrThreadsW);
    const int tj = wtj * (1 * nrThreadsNrThreadsW) + ttj;
    if (tj < nrThreadsW) {
        const int j = bj * (1 * nrThreadsW) + tj;
        if (j < w) {
            const int oi = h - i - 1;
            const int oj = w - j - 1;
            output[(oj + oi * (1 * w)) * 2 + 0] = input[j + i * (1 * w)];
            output[(oj + oi * (1 * w)) * 2 + 1] = 0.0;
        }
    }
}
