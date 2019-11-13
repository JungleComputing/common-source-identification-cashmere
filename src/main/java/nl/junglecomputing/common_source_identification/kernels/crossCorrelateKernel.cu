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

extern "C" {
    __global__ void crossCorrelateKernel(const int n, float* out, const float* x, const float* y);
}

__global__ void crossCorrelateKernel(const int n, float* out, const float* x, const float* y) {
    const int bi = blockIdx.x;
    const int wti = threadIdx.y;
    const int tti = threadIdx.x;

    const int nrThreadsN = min(512, n);
    const int nrThreadsNrThreadsN = min(32, nrThreadsN);
    const int ti = wti * (1 * nrThreadsNrThreadsN) + tti;
    if (ti < nrThreadsN) {
        const int i = bi * (1 * nrThreadsN) + ti;
      if (i < n) {
	const float x_real = x[i * 2];
	const float x_imag = x[i * 2 + 1];
	const float y_real = y[i * 2];
	const float y_imag = y[i * 2 + 1];
	out[i * 2] = x_real * y_real - x_imag * y_imag;
	out[i*2+1] = x_real * y_imag + x_imag * y_real;
      }
    } 
}
