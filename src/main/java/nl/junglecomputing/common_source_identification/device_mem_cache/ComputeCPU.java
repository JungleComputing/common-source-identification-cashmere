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

import java.io.File;
import java.io.IOException;

/*
 * Compute everything on the CPU.  CPU computations do not make use of the cache.
 */
class ComputeCPU {

    static void computeCorrelations(Correlations correlations, int h, int w, File[] imageFiles, int startI, int endI, int startJ,
            int endJ, String executor) throws IOException {
        boolean both = startI == startJ;

        float[][] noisePatternsI;
        float[][] noisePatternsJ;

        noisePatternsI = getNoisePatterns(startI, endI, imageFiles, h, w, executor);

        if (both) {
            noisePatternsJ = noisePatternsI;
        } else {
            noisePatternsJ = getNoisePatterns(startJ, endJ, imageFiles, h, w, executor);
        }

        if (both) {
            for (int i = startI; i < endI; i++) {
                for (int j = i + 1; j < endJ; j++) {
                    computeCorrelation(correlations, i, j, noisePatternsI[i - startI], noisePatternsJ[j - startJ], h, w,
                            executor);
                }
            }
        } else {
            for (int i = startI; i < endI; i++) {
                for (int j = startJ; j < endJ; j++) {
                    computeCorrelation(correlations, i, j, noisePatternsI[i - startI], noisePatternsJ[j - startJ], h, w,
                            executor);
                }
            }
        }
    }

    static void computeCorrelation(Correlations correlations, int i, int j, float[] x, float[] y, int h, int w, String executor) {
        Correlation c = new Correlation(i, j);
        c.coefficient = PCEStage.execute(x, y, h, w, executor);
        correlations.add(c);
    }

    static float[][] getNoisePatterns(int start, int end, File[] imageFiles, int h, int w, String executor) throws IOException {
        int nrNoisePatterns = end - start;

        float[][] noisePatterns = new float[nrNoisePatterns][];

        for (int index = start; index < end; index++) {
            int i = index - start;
            noisePatterns[i] = ComputeNoisePattern.computePRNU(index, imageFiles[index], h, w, executor);
        }

        return noisePatterns;
    }
}
