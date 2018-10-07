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

import java.util.ArrayList;
import java.util.Hashtable;

import org.jocl.Pointer;
import org.jocl.Sizeof;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Device;
import ibis.constellation.Constellation;

/*
 * An ExecutorData instance is associated with a specific executor.  Executors that perform the many-core kernels do not share
 * data.  It is not necessary that each kernel has its own piece of device memory; it is possible that an earlier kernel reuses a
 * temporary buffer from a later kernel.  Therefore, we first allocate generic memory that we address with their logical names,
 * logical meaning what this memory means to a specific kernel.
 */
class ExecutorData {

    private static Hashtable<Constellation, ExecutorData> executorDataMap = new Hashtable<Constellation, ExecutorData>();

    private static ArrayList<ExecutorData> nonUsedExecutorData = new ArrayList<ExecutorData>();

    /*
     * At initialization time we don't know yet to which executors this data will mapped, therefore, we store it in a list and
     * defer creating the mapping to a later stage.
     */
    static synchronized void initialize(int nrExecutors, Device device, int h, int w, int nrBlocksForReduce) {
        for (int i = 0; i < nrExecutors; i++) {
            nonUsedExecutorData.add(new ExecutorData(device, h, w, nrBlocksForReduce));
        }
    }

    /*
     * Get the executor data with a specific executor.  If an executor does not yet have data, we will assign it data.
     */
    static synchronized ExecutorData get(Constellation executor) {
        ExecutorData executorData = executorDataMap.get(executor);
        if (executorData == null) {
            executorData = nonUsedExecutorData.remove(0);
            executorDataMap.put(executor, executorData);
        }
        return executorData;
    }

    /*
     * The amount of device memory that an executor thread needs.
     */
    static long memoryForKernelExecutionThread(int h, int w, int nrBlocksForReduce) {
        return h * w * Sizeof.cl_float * 6L + nrBlocksForReduce * Sizeof.cl_double + nrBlocksForReduce * Sizeof.cl_float
                + (long) Sizeof.cl_int * 5 + h * w * 3;
    }

    private Device device;
    private int nrBlocksForReduce;

    // the allocated memory, generic names
    Pointer memHWFloat1;
    Pointer memHWFloat2;

    Pointer memHWComplex1;
    Pointer memHWComplex2;

    Pointer memDoubleNrBlocksForReduce;
    Pointer memIntNrBlocksForReduce;

    Pointer memInt1;
    Pointer memFloat1;
    Pointer memFloat2;
    Pointer memDouble1;

    Buffer bufferHWRGB;

    // the logical data (where we can choose which actual memory to use)

    // FileToImage
    Buffer imageBuffer;
    Pointer imagePointer;

    // ImageToGrayscale
    // with noisePattern

    // ZeroMean
    // with temp2

    // Wiener
    Pointer complex;
    // with temp2
    Pointer variance;
    Pointer tempFloat;

    // fft
    Pointer temp2;
    // uses data from LeafCorrelationsActivity

    // launchToComplex(AndFlip)
    Pointer noisePattern;
    // uses data from LeafCorrelationsActivity

    // ifft
    Pointer temp;
    // with crossCorrelation

    // FindPeak
    Pointer peak;

    // MaxLocFloats
    Pointer dummyPeak;
    // with indexPeak
    Pointer tempPeaks;
    Pointer indicesPeak;

    // ComputeEnergy
    // with tempEnergy
    Pointer indexPeak;
    Pointer crossCorrelation;

    // SumDoubles
    Pointer tempEnergy;
    Pointer energy;

    // the constructor allocates all the data.
    ExecutorData(Device device, int h, int w, int nrBlocksForReduce) {
        this.device = device;
        this.nrBlocksForReduce = nrBlocksForReduce;

        memHWFloat1 = device.allocate(h * w * Sizeof.cl_float);
        memHWFloat2 = device.allocate(h * w * Sizeof.cl_float);

        memHWComplex1 = device.allocate(h * w * 2 * Sizeof.cl_float);
        memHWComplex2 = device.allocate(h * w * 2 * Sizeof.cl_float);

        memDoubleNrBlocksForReduce = device.allocate(nrBlocksForReduce * Sizeof.cl_double);
        memIntNrBlocksForReduce = device.allocate(nrBlocksForReduce * Sizeof.cl_float);

        memInt1 = device.allocate(Sizeof.cl_int);
        memFloat1 = device.allocate(Sizeof.cl_float);
        memFloat2 = device.allocate(Sizeof.cl_float);
        memDouble1 = device.allocate(Sizeof.cl_double);

        bufferHWRGB = new Buffer(h * w * 3);

        // ImageToGrayscale
        imageBuffer = bufferHWRGB;
        imagePointer = memHWFloat1;

        // Wiener
        complex = memHWComplex1;
        variance = memFloat1;
        tempFloat = memHWFloat1;

        // fft
        temp2 = memHWComplex2;
        // usesdata from LeafCorrelationsActivity

        // launchToComplex(AndFlip)
        noisePattern = memHWFloat2;
        // uses data from LeafCorrelationsActivity

        // ifft
        // with crossCorrelation
        temp = memHWComplex2;

        // FindPeak
        peak = memFloat2;
        // with tempPeaks
        // with indicesPeak
        // with crossCorrelation

        // MaxLocFloats
        dummyPeak = memFloat1;
        // with indexPeak
        tempPeaks = memDoubleNrBlocksForReduce;
        indicesPeak = memIntNrBlocksForReduce;

        // ComputeEnergy
        // with tempEnergy
        indexPeak = memInt1;
        crossCorrelation = memHWComplex1;

        // SumDoubles
        tempEnergy = memDoubleNrBlocksForReduce;
        energy = memDouble1;
    }
}