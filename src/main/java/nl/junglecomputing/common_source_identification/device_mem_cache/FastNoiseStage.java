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

package nl.junglecomputing.common_source_identification.device_mem_cache;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.constellation.Timer;

class FastNoiseStage extends Stage {

    private static final float EPS = 1.0f;

    /**
     * Vertically computes a local gradient for each pixel in an image. Takes forward differences for first and last row. Takes
     * centered differences for interior points.
     *
     * @param h
     *            the image height in pixels
     * @param w
     *            the image width in pixels
     * @param output
     *            the local gradient values
     * @param input
     *            the input image stored as an 1D float array
     */
    static void convolveVertically(int h, int w, float[] output, float[] input) {
        for (int j = 0; j < w; j++) {
            output[0 * w + j] += input[1 * w + j] - input[0 * w + j];
            output[(h - 1) * w + j] += input[(h - 1) * w + j] - input[(h - 2) * w + j];

            for (int i = 1; i < h - 1; i++) {
                output[i * w + j] += 0.5f * (input[(i + 1) * w + j] - input[(i - 1) * w + j]);
            }
        }
    }

    /**
     * Horizontally computes a local gradient for each pixel in an image. Takes forward differences for first and last element.
     * Takes centered differences for interior points.
     *
     * @param h
     *            the image height in pixels
     * @param w
     *            the image width in pixels
     * @param output
     *            the local gradient values
     * @param input
     *            the input image stored as an 1D float array
     */
    static void convolveHorizontally(int h, int w, float[] output, float[] input) {
        for (int i = 0; i < h; i++) {
            output[i * w + 0] += input[i * w + 1] - input[i * w + 0];
            output[i * w + w - 1] += input[i * w + w - 1] - input[i * w + w - 2];

            for (int j = 1; j < w - 1; j++) {
                output[i * w + j] += 0.5f * (input[i * w + j + 1] - input[i * w + j - 1]);
            }
        }
    }

    /**
     * Normalizes gradient values in place.
     *
     * @param h
     *            the image height in pixels
     * @param w
     *            the image width in pixels
     * @param dxs
     *            an array of gradient values
     * @param dys
     *            an array of gradient values
     */
    static void normalize(int h, int w, float[] dxs, float[] dys) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float dx = dxs[i * w + j];
                float dy = dys[i * w + j];

                float norm = (float) Math.sqrt((dx * dx) + (dy * dy));
                float scale = 1.0f / (EPS + norm);

                dxs[i * w + j] = scale * dx;
                dys[i * w + j] = scale * dy;
            }
        }
    }

    /**
     * Zeros all values in an 1D array of float values.
     *
     * @param h
     *            the image height in pixels
     * @param w
     *            the image width in pixels
     * @param input
     *            the array containing only zero values after this method
     */
    static void toZero(int h, int w, float[] input) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                input[i * w + j] = 0.0f;
            }
        }
    }

    /**
     * Applies the FastNoise Filter to extract a PRNU Noise pattern from the input image.
     *
     * @param input
     *            a float array containing a grayscale image or single color channel
     * @return a float array containing the extract PRNU Noise pattern
     */
    public static float[] execute(int h, int w, float[] input, String executor, float[] dxs, float[] dys) {
        Timer timer = Cashmere.getTimer("java", executor, "fastnoise");

        int event;

        event = timer.start();
        Util.zeroArray(dxs);
        timer.stop(event);

        event = timer.start();
        Util.zeroArray(dys);
        timer.stop(event);

        event = timer.start();
        convolveHorizontally(h, w, dxs, input);
        timer.stop(event);

        event = timer.start();
        convolveVertically(h, w, dys, input);
        timer.stop(event);

        event = timer.start();
        normalize(h, w, dxs, dys);
        timer.stop(event);

        event = timer.start();
        toZero(h, w, input);
        timer.stop(event);

        event = timer.start();
        convolveHorizontally(h, w, input, dxs);
        timer.stop(event);

        event = timer.start();
        convolveVertically(h, w, input, dys);
        timer.stop(event);

        return input;
    }

    public static void executeMC(Device device, int h, int w, String executor, KernelLaunch fn1KL, KernelLaunch fn2KL,
            ExecutorData data) throws CashmereNotAvailable {

        MCL.launchFastnoise1Kernel(fn1KL, h, w, data.temp2, false, data.noisePattern, false);
        MCL.launchFastnoise2Kernel(fn2KL, h, w, data.noisePattern, false, data.temp2, false);
    }
}
