/*
 *
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

package nl.junglecomputing.common_source_identification.main_mem_cache;

import java.io.File;
import java.io.IOException;

import ibis.cashmere.constellation.Pointer;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;

import nl.junglecomputing.common_source_identification.cpu.Correlation;

import nl.junglecomputing.common_source_identification.mc.ExecutorData;
import nl.junglecomputing.common_source_identification.mc.ComputeFrequencyDomain;
import nl.junglecomputing.common_source_identification.mc.ComputeCorrelation;
import nl.junglecomputing.common_source_identification.mc.ComputeNoisePattern;

public class CorrelationsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static boolean FLIPPED = true;
    static boolean REGULAR = false;

    public static final String LABEL = "Correlation";
    private ActivityIdentifier id;
    private int height;
    private int width;
    private int i;
    private int j;
    private File fi;
    private File fj;

    public CorrelationsActivity(ActivityIdentifier id, int height, int width, int i, int j, File fi, File fj) {
        super(new Context(LABEL, 1), false);
        this.id = id;
        this.height = height;
        this.width = width;
        this.i = i;
        this.j = j;
        this.fi = fi;
        this.fj = fj;
    }

    @Override
    public int initialize(Constellation constellation) {
        Correlation c = null;
        String executor = constellation.identifier().toString();
	ExecutorData data = ExecutorData.get(constellation);
	c = new Correlation(i, j);
	try {
	    Device device = Cashmere.getDevice("grayscaleKernel");
	    while (!noisePatternOnDevice(executor, i, fi, device, data)) {
		// sleep???
	    }
	    ComputeFrequencyDomain.computeFreq(device, data.noisePatternFreq1, height, width, REGULAR, executor, data);
	    while (!noisePatternOnDevice(executor, j, fj, device, data)) {
		// sleep???
	    }
	    ComputeFrequencyDomain.computeFreq(device, data.noisePatternFreq2, height, width, FLIPPED, executor, data);
	    c.coefficient = ComputeCorrelation.correlateMC(i, j, data.noisePatternFreq1, data.noisePatternFreq2, height,
		    width, executor, device, data);
	} catch (Exception e) {
	    throw new Error(e);
	}
        constellation.send(new Event(identifier(), id, c));
        return FINISH;
    }

    /*
     * This method tries to get the time domain noise pattern on the device and returns true if this succeeds.  If the noise
     * pattern is in the cache (this is the cache of time domain noise patterns that resides in the main memory of the node), we
     * obtain a read lock and we can copy the noise pattern to the device.  If we obtain a write lock, the noise pattern is not
     * in the cache, we have to produce it, put it in the cache, and transfer it to the device.
     *
     *pre:
     * post:
     * - true:
     *   - noise pattern index is on the device
     *   - noise pattern index is in the cache
     * - false
     *   - noise pattern index is not on the device
     *   - noise pattern index is not in the cache
     */
    private boolean noisePatternOnDevice(String thread, int index, File imageFile, Device device, ExecutorData data)
            throws CashmereNotAvailable, LibFuncNotAvailable, IOException {
        try {
            LockToken<Buffer> lt = NoisePatternCache.lockNoisePattern(index);
            if (lt.readLock()) {
                copyNoisePatternToDevice(device, lt.availableElement, data.noisePattern);
                NoisePatternCache.unlockNoisePattern(index);
                return true;
            } else if (lt.writeLock()) {
                produceNoisePattern(thread, index, imageFile, device, lt.availableElement, data);
                NoisePatternCache.unlockNoisePattern(index);
                return true;
            } else {
                throw new Error("should not happen");
            }
        } catch (LockException e) {
            return false;
        }
    }

    /*
     * pre:
     * - the current thread holds a read lock for noisePattern
     * post:
     * - the noisePattern is on the device
     */
    private void copyNoisePatternToDevice(Device device, Buffer noisePattern, Pointer devicenp) {
        device.copy(noisePattern, devicenp);
    }

    /*
     * pre:
     * - the current thread holds a write lock for noise pattern index
     * post:
     * - noise pattern index is on device
     * - noise pattern index is in the cache
    */
    private void produceNoisePattern(String thread, int index, File imageFile, Device device, Buffer noisePattern,
            ExecutorData data) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        ComputeNoisePattern.computePRNU_MC(index, imageFile, height, width, thread, device, data);
        // get the data from the device, noisePattern points to memory in the cache
        device.get(noisePattern, data.noisePattern);
    }

    @Override
    public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // empty
    }

}
