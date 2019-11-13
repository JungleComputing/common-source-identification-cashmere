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
    __global__ void zeromeanVerticallyKernel(const int h, const int w, float* output, const float* input);
}

__global__ void zeromeanVerticallyKernel(const int h, const int w, float* output, const float* input) {
    const int bj = blockIdx.x;
    const int wtj = threadIdx.y;
    const int ttj = threadIdx.x;

    const int nrThreadsW = min(1024, w);
    const int nrThreadsNrThreadsW = min(32, nrThreadsW);
    const int tj = wtj * (1 * nrThreadsNrThreadsW) + ttj;
    if (tj < nrThreadsW) {
        const int j = bj * (1 * nrThreadsW) + tj;
        if (j < w) {
            float sumEven = 0.0;
            float sumOdd = 0.0;
            for (int i = 0; i < h - 1; i += 2) {
                sumEven += input[j + i * (1 * w)];
                sumOdd += input[j + (i + 1) * (1 * w)];
            }
            const float meanEven = sumEven / ((h + 1) / 2);
            const float meanOdd = sumOdd / (h / 2);
            for (int i = 0; i < h - 1; i += 2) {
                output[j + i * (1 * w)] = input[j + i * (1 * w)] - meanEven;
                output[j + (i + 1) * (1 * w)] = input[j + (i + 1) * (1 * w)] - meanOdd;
            }
        }
    }
}
