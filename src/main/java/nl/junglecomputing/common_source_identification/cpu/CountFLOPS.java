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

package nl.junglecomputing.common_source_identification.cpu;

import java.util.HashMap;
import java.util.Map;

public class CountFLOPS {

    static final Map<ImageDims, Long> FFT_FLOPS_FORWARD = createFFTFlopsMap(true);
    static final Map<ImageDims, Long> FFT_FLOPS_BACKWARD = createFFTFlopsMap(false);

    /*
     * Functions for images and results
     */
    static Map<ImageDims, Long> createFFTFlopsMap(boolean forward) {
        HashMap<ImageDims, Long> map = new HashMap<ImageDims, Long>();
        if (forward) {
            map.put(new ImageDims(3000, 4000), 2852200000l);
            map.put(new ImageDims(2736, 3648), 1798947504l);
        } else {
            map.put(new ImageDims(3000, 4000), 2852200000l);
            map.put(new ImageDims(2736, 3648), 1798947504l);

        }
        return map;
    }

    static long nrFlopsGrayscale(int h, int w) {
        return 5 * h * w;
    }

    static long nrFlopsFastNoise(int h, int w) {
        return 2 * 3 * h * w;
    }

    static long nrFlopsZeroMean(int h, int w) {
        return 2 * (h * w * 2 + 2 * w);
    }

    static long nrFlopsWiener(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return flopsForwardFFT + h * w * 3 + h * w * 176 + (h * w * 3 + 2) + h * w * 4 + flopsBackwardFFT;
    }

    static long nrFlopsPRNU(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return nrFlopsGrayscale(h, w) + nrFlopsFastNoise(h, w) + nrFlopsZeroMean(h, w)
                + nrFlopsWiener(h, w, flopsForwardFFT, flopsBackwardFFT);
    }

    static long nrFlopsPCELinear(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return flopsForwardFFT * 2;
    }

    static long nrFlopsPCEQuadratic(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return h * w * 6 + flopsBackwardFFT + h * w + (h - 11) * (h - 11) * 2 + 2;
    }

    static long nrFlops(int h, int w, int n, int nrNoisePatternsComputed, int nrNoisePatternFreqTransforms) {
        ImageDims imageDims = new ImageDims(h, w);
        Long flopsForwardFFT = FFT_FLOPS_FORWARD.get(imageDims);
        Long flopsBackwardFFT = FFT_FLOPS_BACKWARD.get(imageDims);

        if (flopsForwardFFT == null || flopsBackwardFFT == null) {
            return 0;
        }

        long flopsPRNU = nrFlopsPRNU(h, w, flopsForwardFFT, flopsBackwardFFT);
        long flopsPCELinear = nrFlopsPCELinear(h, w, flopsForwardFFT, flopsBackwardFFT);
        long flopsPCEQuadratic = nrFlopsPCEQuadratic(h, w, flopsForwardFFT, flopsBackwardFFT);

        return nrNoisePatternsComputed * flopsPRNU + nrNoisePatternFreqTransforms * flopsPCELinear
                + ((n * (n - 1)) / 2) * flopsPCEQuadratic;
    }

    public static void printGFLOPS(int h, int w, int n, int nrNoisePatternsComputed, int nrNoisePatternFreqTransforms,
            long timeNanos) {
        long nrFlopsAchieved = nrFlops(h, w, n, nrNoisePatternsComputed, nrNoisePatternFreqTransforms);
        long nrFlopsActual = nrFlops(h, w, n, n, n);
        double timeSeconds = timeNanos / 1e9;

        System.out.printf("achieved performance (counting everything computed): %.2f GFLOPS\n",
                nrFlopsAchieved / timeSeconds / 1e9);
        System.out.printf("actual performance (counting only what should be computed): %.2f GFLOPS\n",
                nrFlopsActual / timeSeconds / 1e9);
    }
}
