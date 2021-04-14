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

import org.slf4j.Logger;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.constellation.util.MemorySizes;
// import sun.misc.VM; not possible since Java 9
import nl.junglecomputing.common_source_identification.GetMaxDirectMem;

@SuppressWarnings("restriction")
public class CacheConfig {

    static Logger logger = CommonSourceIdentification.logger;

    public static int nrNoisePatternsForSpace(long space, long sizeNoisePattern) {
        return (int) Math.floor(space / (double) sizeNoisePattern);
    }

    // get the number of noise patterns that the many-core device can hold
    public static int getNrNoisePatternsDevice(long sizeNoisePattern, long toBeReserved) {
        try {
            Device device = Cashmere.getDevice("grayscaleKernel");
            long spaceDevice = device.getMemoryCapacity();
            long spaceForNoisePatterns = spaceDevice - toBeReserved - 500 * MemorySizes.MB;

            int nrNoisePatterns = nrNoisePatternsForSpace(spaceForNoisePatterns, sizeNoisePattern);

            if (logger.isDebugEnabled()) {
                logger.debug("device memory: " + MemorySizes.toStringBytes(spaceDevice));
                logger.debug("to be reserved: " + MemorySizes.toStringBytes(toBeReserved));
                logger.debug("space for patterns on device: " + MemorySizes.toStringBytes(spaceForNoisePatterns));
                logger.debug("device holds a maximum of {} noise patterns", nrNoisePatterns);
            }

            return nrNoisePatterns;
        } catch (CashmereNotAvailable e) {
            throw new Error(e);
        }
    }

    public static int getNrNoisePatternsMemory(int sizeNoisePattern, long spaceForGrayscale) {
        // long spaceForNoisePatterns = VM.maxDirectMemory() - spaceForGrayscale;
        long spaceForNoisePatterns = GetMaxDirectMem.maxDirectMemory() - spaceForGrayscale;
        int nrNoisePatterns = nrNoisePatternsForSpace(spaceForNoisePatterns, sizeNoisePattern);

        if (logger.isDebugEnabled()) {
            logger.debug("space for noise patterns: " + MemorySizes.toStringBytes(spaceForNoisePatterns));
            logger.debug(String.format("The memory will hold a maximum of " + "%d noise patterns", nrNoisePatterns));
        }

        return nrNoisePatterns;
    }

    public static int initializeCache(Device device, int height, int width, long toBeReserved, int nrThreads) {
        int sizeNoisePattern = height * width * 4;
        int sizeNoisePatternFreq = sizeNoisePattern * 2;
        if (logger.isDebugEnabled()) {
            logger.debug("Size of noise pattern: " + MemorySizes.toStringBytes(sizeNoisePattern));
            logger.debug("Size of noise pattern freq: " + MemorySizes.toStringBytes(sizeNoisePatternFreq));
        }

        int nrNoisePatternsFreqDevice = getNrNoisePatternsDevice(sizeNoisePatternFreq, toBeReserved);
        int memReservedForGrayscale = height * width * 3 * nrThreads;
        int nrNoisePatternsMemory = getNrNoisePatternsMemory(sizeNoisePattern, memReservedForGrayscale);

        NoisePatternCache.initialize(device, height, width, nrNoisePatternsFreqDevice, nrNoisePatternsMemory);

        return nrNoisePatternsFreqDevice;
    }

    public static void clearCaches() {
        NoisePatternCache.clear();
    }
}
