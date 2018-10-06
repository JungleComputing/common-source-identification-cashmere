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

import ibis.cashmere.constellation.Cashmere;
import ibis.constellation.Timer;

/*
 * In this stage we compute the Peak-To-Correlation Energy of two noise patterns in the time domain
 */

class PCEStage extends Stage {

    static final int SQUARE_SIZE = 11;

    static double execute(float[] x, float[] y, int h, int w, String executor) {
        Timer timer = Cashmere.getTimer("java", executor, "pce");
        Timer fftTimer = Cashmere.getTimer("java", executor, "fft");

        int event;

        event = timer.start();
        float[] complexY = toComplexAndFlip(y, h, w, true);
        timer.stop(event);

        event = fftTimer.start();
        Util.fft(h, w, complexY);
        fftTimer.stop(event);

        event = timer.start();
        float[] complexX = toComplexAndFlip(x, h, w, false);
        timer.stop(event);

        event = fftTimer.start();
        Util.fft(h, w, complexX);
        fftTimer.stop(event);

        event = timer.start();
        float[] crossCorrelation = computeCrossCorrelation(complexX, complexY);
        timer.stop(event);

        event = fftTimer.start();
        Util.ifft(h, w, crossCorrelation);
        fftTimer.stop(event);

        event = timer.start();
        int peakIndex = findPeak(crossCorrelation);

        double peak = crossCorrelation[((h * w) - 1) << 1];
        int indexY = peakIndex / w;
        int indexX = peakIndex - (indexY * w);
        timer.stop(event);

        event = timer.start();
        double energy = energyFixed(h, w, crossCorrelation, SQUARE_SIZE, indexX, indexY);
        double absPce = (peak * peak) / energy;

        timer.stop(event);

        return absPce;
    }

    static float[] toComplexAndFlip(float[] fs, int h, int w, boolean flip) {
        float[] complex = new float[h * w * 2];

        float[] row = new float[w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                row[j] = fs[i * w + j];
            }

            int offset = flip ? (((h - i) * w) - 1) * 2 : i * w * 2;

            for (int j = 0; j < w; j++) {
                if (flip) {
                    complex[offset - 2 * j] = row[j];
                    complex[offset - 2 * j + 1] = 0.0f;
                } else {
                    complex[offset + 2 * j] = row[j];
                    complex[offset + 2 * j + 1] = 0.0f;
                }
            }
        }
        return complex;
    }

    static float[] computeCrossCorrelation(float[] x, float[] y) {
        float[] crossCorrelation = new float[x.length];

        for (int i = 0; i < x.length; i += 2) {
            float xRe = x[i];
            float xIm = x[i + 1];
            float yRe = y[i];
            float yIm = y[i + 1];
            crossCorrelation[i] = (xRe * yRe) - (xIm * yIm);
            crossCorrelation[i + 1] = (xRe * yIm) + (xIm * yRe);
        }

        return crossCorrelation;
    }

    static int findPeak(float[] crossCorrelation) {
        float max = 0.0f;
        int res = 0;
        for (int i = 0; i < crossCorrelation.length; i += 2) {
            if (Math.abs(crossCorrelation[i]) > max) {
                max = Math.abs(crossCorrelation[i]);
                res = i;
            }
        }
        // divided by 2, because we want the index of the number in the complex
        // array
        return res / 2;
    }

    static double energyFixed(int h, int w, float[] crossCorrelation, int squareSize, int peakIndexX, int peakIndexY) {
        int radius = (squareSize - 1) / 2;
        int n = (h * w) - (squareSize * squareSize);

        // Determine the energy, i.e. the sample variance of circular
        // cross-correlations, minus an area around the peak
        double energy = 0.0;
        for (int row = 0; row < h; row++) {
            boolean peakRow = row > peakIndexY - radius && row < peakIndexY + radius;
            for (int col = 0; col < w; col++) {
                if (peakRow && col > peakIndexX - radius && col < peakIndexX + radius) {
                    continue;
                } else {
                    float f = crossCorrelation[row * w * 2 + col * 2];
                    energy += (f * f);
                }
            }
        }
        return energy / n;
    }
}
