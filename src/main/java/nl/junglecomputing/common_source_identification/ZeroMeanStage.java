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

package nl.junglecomputing.common_source_identification;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.constellation.Timer;

/*
 * Stage that applies the ZeroMean filter
 */
public class ZeroMeanStage extends Stage {

    /**
     * Applies an in place zero mean filtering operation to each column in an image. First two mean values are computed, one for
     * even and one for odd elements, for each column in the image. Then, the corresponding mean value is subtracted from each
     * pixel value in the image.
     *
     * @param h
     *            - the image height in pixels
     * @param w
     *            - the image width in pixels
     * @param input
     *            - the image stored as a 1D array of float values
     */
    static void computeMeanVertically(int h, int w, float[] input) {
        for (int j = 0; j < w; j++) {
            float sumEven = 0.0f;
            float sumOdd = 0.0f;

            for (int i = 0; i < h - 1; i += 2) {
                sumEven += input[i * w + j];
                sumOdd += input[(i + 1) * w + j];
            }
            if (!isDivisibleByTwo(h)) {
                sumEven += input[(h - 1) * w + j];
            }

            float meanEven = sumEven / ((h + 1) / 2);
            float meanOdd = sumOdd / (h / 2);

            for (int i = 0; i < h - 1; i += 2) {
                input[i * w + j] -= meanEven;
                input[(i + 1) * w + j] -= meanOdd;
            }
            if (!isDivisibleByTwo(h)) {
                input[(h - 1) * w + j] -= meanEven;
            }
        }
    }

    private static boolean isDivisibleByTwo(final int value) {
        return (value & 1) == 0;
    }

    /**
     * Applies the Zero Mean Total filter on the CPU. This routine is mainly here for comparing the result and performance with
     * the GPU version.
     *
     * @param input
     *            - the image stored as a 1D array of float values
     * @return - the filtered image as a 1D float array
     */
    public static float[] execute(int h, int w, float[] input, String executor) {
        Timer timer = Cashmere.getTimer("java", executor, "zeromean");
        int event;

        event = timer.start();
        computeMeanVertically(h, w, input);
        timer.stop(event);

        event = timer.start();
        float[] inputTransposed = Util.transpose(h, w, input);
        timer.stop(event);

        event = timer.start();
        computeMeanVertically(w, h, inputTransposed);
        timer.stop(event);

        event = timer.start();
        float[] output = Util.transpose(w, h, inputTransposed);
        timer.stop(event);

        return output;
    }

    public static void executeMC(Device device, int h, int w, String executor, ExecutorData data) throws CashmereNotAvailable {

        Timer timer = Cashmere.getTimer("GPU", executor, "zeromean");
        int event = timer.start();

        Kernel verKernel = Cashmere.getKernel("zeromeanVerticallyKernel", device);
        Kernel traKernel = Cashmere.getKernel("transposeKernel", device);

        KernelLaunch ver1KL = verKernel.createLaunch();
        KernelLaunch tra1KL = traKernel.createLaunch();
        KernelLaunch ver2KL = verKernel.createLaunch();
        KernelLaunch tra2KL = traKernel.createLaunch();

        if (logger.isDebugEnabled()) {
            logger.debug("Executing the zeromean stage");
        }

        MCL.launchZeromeanVerticallyKernel(ver1KL, h, w, data.temp2, false, data.noisePattern, false);
        MCL.launchTransposeKernel(tra1KL, w, h, data.noisePattern, false, data.temp2, false);
        MCL.launchZeromeanVerticallyKernel(ver2KL, w, h, data.temp2, false, data.noisePattern, false);
        MCL.launchTransposeKernel(tra2KL, h, w, data.noisePattern, false, data.temp2, false);

        timer.stop(event);
    }
}
