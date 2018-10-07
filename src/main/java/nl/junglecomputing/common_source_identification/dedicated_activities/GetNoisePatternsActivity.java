package nl.junglecomputing.common_source_identification.dedicated_activities;

import java.io.File;
import java.io.Serializable;
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
import nl.junglecomputing.common_source_identification.remote_activities.LockToken;
import nl.junglecomputing.common_source_identification.remote_activities.NoisePatternCache;

/**
 * This Activity's only task is to wait for requests for time domain noise patterns, compute them, and send them back.
 */
public class GetNoisePatternsActivity extends Activity {

    private static final long serialVersionUID = 1L;
    public static final String LABEL = "GetNP";

    private final int height;
    private final int width;
    private transient String executor;
    private transient Device device;
    private final ActivityIdentifier parent;

    static AtomicInteger countFetched = new AtomicInteger(0);

    public static class PatternsInfo implements Serializable {

        private static final long serialVersionUID = 1L;

        public int[] indices;
        public File[] files;
        public ActivityIdentifier target;
    };

    public GetNoisePatternsActivity(ActivityIdentifier parent, int height, int width, int location) {
        super(new Context(LABEL + location, 1), true);
        this.height = height;
        this.width = width;
        this.parent = parent;
    }

    @Override
    public int initialize(Constellation constellation) {
        executor = constellation.identifier().toString();

        try {
            device = Cashmere.getDevice("grayscaleKernel");
        } catch (CashmereNotAvailable e1) {
            throw new Error("Could not get device", e1);
        }
        constellation.send(new Event(identifier(), parent, "I'm here!"));
        return SUSPEND;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        Object messageData = event.getData();
        if (messageData == null) {
            return FINISH;
        }

        Timer t = Cashmere.getTimer("java", executor, "provide noise pattern");
        int ev = t.start();

        PatternsInfo info = (PatternsInfo) messageData;
        int[] indices = info.indices;
        File[] files = info.files;

        countFetched.addAndGet(indices.length);

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
        constellation.send(new Event(identifier(), info.target, b));
        // Only release the locks after the message is sent.
        for (int i = 0; i < indices.length; i++) {
            NoisePatternCache.unlockNoisePattern(indices[i]);
        }

        t.stop(ev);
        return SUSPEND;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // nothing
    }
}