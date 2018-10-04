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

import org.jocl.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.LibFuncNotAvailable;

class ComputeCorrelation {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.ComputeCorrelation");

    static long timeComputeCorrelations = 0;

    static double correlateMC(int i, int j, Pointer noisePatternI, Pointer noisePatternJ, int h, int w, String executor,
            Device device, ExecutorData data) throws CashmereNotAvailable, LibFuncNotAvailable {

        if (logger.isDebugEnabled()) {
            logger.debug("doing PCE on MC for {} and {}", i, j);
        }

        // reductions are performed with nrBlocksForReduce blocks that in a later stage will be reduced to 1 value
        int nrBlocksForReduce = 1024;

        return PCEStage.executeMC(device, noisePatternI, noisePatternJ, h, w, executor, nrBlocksForReduce, data);
    }
}
