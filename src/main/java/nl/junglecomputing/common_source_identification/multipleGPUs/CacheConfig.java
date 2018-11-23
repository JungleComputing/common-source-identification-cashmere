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

package nl.junglecomputing.common_source_identification.multipleGPUs;

import org.slf4j.Logger;

import ibis.constellation.util.MemorySizes;
import sun.misc.VM;

@SuppressWarnings("restriction")
public class CacheConfig {

    static Logger logger = CommonSourceIdentification.logger;

    private static int nrNoisePatternsForSpace(long space, long sizeNoisePattern) {
        return (int) Math.floor(space / (double) sizeNoisePattern);
    }

    public static int getNrNoisePatternsMemory(int sizeNoisePattern, long spaceForGrayscale) {
        long spaceForNoisePatterns = VM.maxDirectMemory() - spaceForGrayscale;
        int nrNoisePatterns = nrNoisePatternsForSpace(spaceForNoisePatterns, sizeNoisePattern);

        if (logger.isDebugEnabled()) {
            logger.debug("space for noise patterns: " + MemorySizes.toStringBytes(spaceForNoisePatterns));
            logger.debug(String.format("The memory will hold a maximum of " + "%d noise patterns", nrNoisePatterns));
        }

        return nrNoisePatterns;
    }
}
