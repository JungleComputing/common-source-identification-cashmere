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

package nl.junglecomputing.common_source_identification.mc;

import org.jocl.cl_command_queue;
import org.jocl.cl_event;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.LibFunc;
import ibis.cashmere.constellation.LibFuncLaunch;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Timer;

/*
 * Stage that applies the Wiener filter.
 */
class WienerStage extends Stage {

    static final int[] FILTER_SIZES = { 3, 5, 7, 9 };

    /**
     * Computes the square of each frequency and stores the result as a real.
     *
     * @param h
     *            - the image height in pixels
     * @param w
     *            - the image width in pixels
     * @param frequencies
     *            - the frequencies as the result of Fourier transform
     * @return - a float array containing the frequencies squared as real values
     */
    public static float[] computeSquaredMagnitudes(int h, int w, float[] frequencies) {
        float[] result = new float[h * w];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float re = frequencies[i * 2 * w + (2 * j)];
                float im = frequencies[i * 2 * w + (2 * j + 1)];
                result[i * w + j] = (re * re) + (im * im);
            }
        }

        return result;
    }

    /**
     * This function scales the frequencies in input with a combination of the global variance and an estimate for the local
     * variance at that position. Effectively this cleans the input pattern from low frequency noise.
     *
     * @param h
     *            - the image height in pixels
     * @param w
     *            - the image width in pixels
     * @param input
     *            - a float array of complex values that contain the frequencies
     * @param varianceEstimates
     *            - an array containing the estimated local variance
     * @param variance
     *            - the global variance of the input
     * @return - a complex array in which the frequencies have been scaled
     */
    static float[] scaleWithVariances(int h, int w, float[] input, float[] varianceEstimates, float variance) {
        float[] output = new float[h * w * 2];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float scale = variance / Math.max(variance, varianceEstimates[i * w + j]);
                output[i * (2 * w) + (j * 2)] = input[i * (2 * w) + (j * 2)] * scale;
                output[i * (2 * w) + (j * 2 + 1)] = input[i * (2 * w) + (j * 2 + 1)] * scale;
            }
        }

        return output;
    }

    /**
     * Estimates the minimum local variances by applying all filters.
     *
     * @param h
     *            - the image height in pixels
     * @param w
     *            - the image width in pixels
     * @param squaredMagnitudes
     *            - the input array containing the squared frequency values as reals
     * @return - a float array containing the estimated minimum local variance
     */
    static float[] computeVarianceEstimates(int h, int w, float[] squaredMagnitudes) {

        float[] varianceEstimates = Util.from2DTo1D(h, w, Util.initializeArray(h, w, Float.MAX_VALUE));
        for (final int filterSize : FILTER_SIZES) {
            float[] squaredMagnitudesWithBorder = Util.addBorder(h, w, squaredMagnitudes, filterSize / 2);
            float[] output = Util.convolve(h, w, filterSize, squaredMagnitudesWithBorder);
            varianceEstimates = Util.minimum(varianceEstimates, output);
        }

        return varianceEstimates;
    }

    /**
     * Applies the Wiener Filter to the input pattern on the CPU. This function is mainly used to check the GPU result.
     *
     * @param input
     *            - the input pattern stored as an 1D float array
     * @return - a float array containing the filtered pattern
     */
    public static float[] execute(int h, int w, float[] input, String executor) {
        Timer timer = Cashmere.getTimer("java", executor, "wiener");
        Timer fftTimer = Cashmere.getTimer("java", executor, "fft");

        int event;

        event = timer.start();
        // convert input to complex values
        float[] complex = Util.toComplex(h, w, input);
        timer.stop(event);

        // forward Fourier transform using JTransforms
        event = fftTimer.start();
        Util.fft(h, w, complex);
        fftTimer.stop(event);

        // compute frequencies squared and store as real
        event = timer.start();
        float[] squaredMagnitudes = computeSquaredMagnitudes(h, w, complex);
        timer.stop(event);

        // estimate local variances and keep the mimimum
        event = timer.start();
        float[] varianceEstimates = computeVarianceEstimates(h, w, squaredMagnitudes);
        timer.stop(event);

        // compute global variance, assuming zero mean
        int n = w * h;
        event = timer.start();
        float variance = (Util.sum(Util.multiply(input, input)) * n) / (n - 1);
        timer.stop(event);

        // scale the frequencies with the global and local variance
        event = timer.start();
        float[] frequenciesScaled = scaleWithVariances(h, w, complex, varianceEstimates, variance);
        timer.stop(event);

        // inverse Fourier transform
        event = fftTimer.start();
        Util.ifft(h, w, frequenciesScaled);
        fftTimer.stop(event);

        // convert values to real and return result
        event = timer.start();
        float[] result = Util.assign(input, Util.toReal(frequenciesScaled));
        timer.stop(event);

        return result;
    }

    public static void executeMC(Device device, int h, int w, String executor, ExecutorData data)
            throws CashmereNotAvailable, LibFuncNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "wiener");

        int tevent = timer.start();

        // convert input to complex values
        Kernel comKernel = Cashmere.getKernel("toComplexKernel", device);

        KernelLaunch comKL = comKernel.createLaunch();

        if (logger.isDebugEnabled()) {
            logger.debug("Executing the Wiener stage");
        }

        MCL.launchToComplexKernel(comKL, h * w, data.complex, false, data.noisePattern, false);

        Kernel vzmKernel = Cashmere.getKernel("varianceZeroMeanKernel", device);

        KernelLaunch vzmKL = vzmKernel.createLaunch();

        MCL.launchVarianceZeroMeanKernel(vzmKL, h * w, data.variance, false, data.noisePattern, false);

        // forward fft
        LibFunc fft = Cashmere.getLibFunc("fft", device);
        LibFuncLaunch fftLaunch = fft.createLaunch();
        fftLaunch.launch((cl_command_queue queue, int num_events_in_wait_list, cl_event[] event_wait_list, cl_event event) -> FFT
                .performFFT(queue, h, w, data.complex, data.temp2, true, num_events_in_wait_list, event_wait_list, event));

        Kernel sqMKernel = Cashmere.getKernel("computeSquaredMagnitudesKernel", device);
        KernelLaunch sqKL = sqMKernel.createLaunch();

        MCL.launchComputeSquaredMagnitudesKernel(sqKL, h, w, data.noisePattern, false, data.complex, false);

        Kernel veKernel = Cashmere.getKernel("computeVarianceEstimatesKernel", device);

        KernelLaunch veKL = veKernel.createLaunch();

        MCL.launchComputeVarianceEstimatesKernel(veKL, h, w, data.tempFloat, false, data.noisePattern, false);

        // scale the frequencies with the global and local variance
        Kernel fsKernel = Cashmere.getKernel("scaleWithVariancesKernel", device);

        KernelLaunch fsKL = fsKernel.createLaunch();

        MCL.launchScaleWithVariancesKernel(fsKL, h, w, data.complex, false, data.tempFloat, false, data.variance, false);

        // inverse Fourier transform
        LibFuncLaunch ifftLaunch = fft.createLaunch();
        ifftLaunch.launch((cl_command_queue queue, int numEventsWaitList, cl_event[] event_wait_list, cl_event event) -> FFT
                .performFFT(queue, h, w, data.complex, data.temp2, false, numEventsWaitList, event_wait_list, event));

        // convert values to real and return result

        Kernel trKernel = Cashmere.getKernel("toRealKernel", device);

        KernelLaunch trKL = trKernel.createLaunch();

        MCL.launchToRealKernel(trKL, h * w, data.noisePattern, false, data.complex, false);

        timer.stop(tevent);
    }
}
