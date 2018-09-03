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

import java.io.File;
import java.io.IOException;

/*
 * Compute everything on the CPU.  CPU computations do not make use of the cache.
 */
class ComputeCPU {

    static Correlation computeCorrelation(int h, int w, int i, int j, File fi, File fj, String executor) throws IOException {

        float[] noisePatternI = ComputeNoisePattern.computePRNU(i, fi, h, w, executor);
        float[] noisePatternJ = ComputeNoisePattern.computePRNU(j, fj, h, w, executor);

        Correlation c = new Correlation(i, j);
        c.coefficient = PCEStage.execute(noisePatternI, noisePatternJ, h, w, executor);
        return c;
    }
}
