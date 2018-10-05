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

#define EPS (1.0)






__kernel void fastnoise1Kernel(const int h, const int w, __global float* dxsdys, const __global float* input) {
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
            float dx;
            if (j == 0) dx = input[j + 1 + i * (1 * w)] - input[j + i * (1 * w)];
            else if (j == w - 1) dx = input[j + i * (1 * w)] - input[j - 1 + i * (1 * w)];
            else dx = 0.5 * (input[j + 1 + i * (1 * w)] - input[j - 1 + i * (1 * w)]);
            float dy;
            if (i == 0) dy = input[j + (i + 1) * (1 * w)] - input[j + i * (1 * w)];
            else if (i == h - 1) dy = input[j + i * (1 * w)] - input[j + (i - 1) * (1 * w)];
            else dy = 0.5 * (input[j + (i + 1) * (1 * w)] - input[j + (i - 1) * (1 * w)]);
            const float norm = sqrt(dx * dx + dy * dy);
            const float scale = 1.0 / (EPS + norm);
            dxsdys[j + 0 * (1 * h * w) + i * (1 * w)] = dx * scale;
            dxsdys[j + 1 * (1 * h * w) + i * (1 * w)] = dy * scale;
        }
    }
}
