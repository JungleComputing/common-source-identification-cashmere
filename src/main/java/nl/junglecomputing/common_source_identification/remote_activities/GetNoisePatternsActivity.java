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

package nl.junglecomputing.common_source_identification.remote_activities;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.Timer;
import nl.junglecomputing.common_source_identification.main_mem_cache.LockException;
import nl.junglecomputing.common_source_identification.mc.ComputeNoisePattern;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;

class GetNoisePatternsActivity extends Activity {

    private static final long serialVersionUID = 1L;
    static final String LABEL = "GetNP";

    private final ActivityIdentifier parent;
    private final int[] indices;
    private final File[] files;
    private final int height;
    private final int width;

    public static AtomicInteger countFetched = new AtomicInteger(0);

    public GetNoisePatternsActivity(ActivityIdentifier id, File[] files, int height, int width, int[] indices, int loc_indices) {
        super(new Context(LABEL + loc_indices, 1), false);
        this.parent = id;
        this.indices = indices;
        this.files = files;
        this.height = height;
        this.width = width;
    }

    @Override
    public int initialize(Constellation constellation) {
        String executor = constellation.identifier().toString();

        countFetched.addAndGet(indices.length);

        Timer t = Cashmere.getTimer("java", executor, "provide noise pattern");
        int ev = t.start();

        Device device;
        try {
            device = Cashmere.getDevice("grayscaleKernel");
        } catch (CashmereNotAvailable e1) {
            throw new Error("Could not get device", e1);
        }

        Buffer[] bufs = new Buffer[indices.length];
        ArrayList<LockToken<Buffer>> locks = new ArrayList<LockToken<Buffer>>();
        for (int i = 0; i < indices.length; i++) {
            locks.add(null);
        }

        boolean done = false;
        while (!done) {
            done = true;
            for (int i = 0; i < indices.length; i++) {
                LockToken<Buffer> lt = locks.get(i);
                if (lt == null) {
                    try {
                        lt = NoisePatternCache.lockNoisePattern(indices[i]);
                        locks.set(i, lt);
                    } catch (LockException e) {
                        done = false;
                        continue;
                    }
                    if (lt.readLock()) {
                        bufs[i] = lt.availableElement;
                    } else {
                        assert (lt.writeLock());
                        // Compute the time domain noise pattern
                        ExecutorData data = ExecutorData.get(constellation);
                        try {
                            ComputeNoisePattern.computePRNU_MC(indices[i], files[i], height, width, executor, device, data);
                            device.get(lt.availableElement, data.noisePattern);
                            NoisePatternCache.toReadLockNoisePattern(indices[i]);
                            bufs[i] = lt.availableElement;
                        } catch (Exception e) {
                            throw new Error(e);
                        }
                    }
                }
            }
        }
        NPBuffer b = new NPBuffer(bufs, indices);
        constellation.send(new Event(identifier(), parent, b));
        // Only release the locks after the message is sent.
        for (int i = 0; i < indices.length; i++) {
            NoisePatternCache.unlockNoisePattern(indices[i]);
        }

        t.stop(ev);
        return FINISH;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // nothing
    }
}
