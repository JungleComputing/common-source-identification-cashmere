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

package nl.junglecomputing.common_source_identification.dedicated_activities;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.jocl.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.Timer;
import ibis.constellation.util.ByteBufferCache;
import nl.junglecomputing.common_source_identification.cpu.Correlation;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;
import nl.junglecomputing.common_source_identification.device_mem_cache.LockToken;
import nl.junglecomputing.common_source_identification.device_mem_cache.NoisePatternCache;
import nl.junglecomputing.common_source_identification.main_mem_cache.LockException;
import nl.junglecomputing.common_source_identification.mc.ComputeCorrelation;
import nl.junglecomputing.common_source_identification.mc.ComputeFrequencyDomain;
import nl.junglecomputing.common_source_identification.mc.ComputeNoisePattern;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;

/*
 * LeafCorelationsActivity is the leaf node in the tree of CorrelationsActivities and performs the actual correlations.  This
 * activity will always execute on the node where it was created.  An instace of this class produces noise patterns, noise
 * patterns in the frequency domain, and the correlations that will be sent to its parent.
 *
 * The LeafCorelationsActivity will produce correlations for an i-range and j-range noise patterns.  The size of those ranges are
 * determined by the Activity that created this one and is based on a threshold based on the amount of memory on the device and
 * the number of Executors.
 *
 * To produce correlations, the noise patterns have to be locked with read locks, if a thread does not get a read lock, but a
 * write lock, it means that the noise pattern has to be produced, which is also done by this class.
 */
class LeafCorrelationsActivity extends CorrelationsActivity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.LeafCorrelationsActivity");
    static Logger memlogger = LoggerFactory.getLogger("CommonSourceIdentification.Cache");

    static boolean FLIPPED = true;
    static boolean REGULAR = false;

    // keeping track of the noise patterns
    private transient Pointer[] noisePatternsXFreq;
    private transient Pointer[] noisePatternsYFreq;

    // The data that belongs to the executor that executes this activity.
    private transient ExecutorData data;

    // keeping track of the time
    private transient int event;
    private transient Timer timer;

    // debugging, keeps track of the amount of the total LeafCorrelationsActivities that are in flight on the node.
    static int inFlight = 0;

    transient boolean[] needed;

    private transient Device device;

    private transient Event message = null;

    private transient Random random;

    /*
     * Create a LeafCorrelationsActivity with the i and j range indicated by indicesI and indicesJ.
     */
    LeafCorrelationsActivity(ActivityIdentifier parent, ActivityIdentifier progressActivityID, int[] indicesI, int[] indicesJ,
            int hostI, int hostJ, File[] filesI, File[] filesJ, int h, int w, int level, ActivityIdentifier[][] providers) {

        // made node2 node1 always equal, to trigger the fact that this should be local.
        super(true, parent, progressActivityID, indicesI, indicesJ, hostI, hostJ, filesI, filesJ, h, w, level, providers);
        logger.debug("Creating LeafCorrelation, node1 = {}, node2 = {}, size1 = {}, size2 = {}", node1, node2, indicesI.length,
                indicesJ.length);
    }

    @Override
    public int initialize(Constellation cons) {
        logger.debug("Running LeafCorrelation, node1 = {}, node2 = {}, size1 = {}, size2 = {}", node1, node2, indicesI.length,
                indicesJ.length);

        this.noisePatternsXFreq = new Pointer[indicesI.length];
        this.noisePatternsYFreq = new Pointer[indicesJ.length];

        random = new Random();

        String executor = cons.identifier().toString();
        this.timer = Cashmere.getTimer("java", executor, "leaf correlations");
        this.event = timer.start();

        // we retrieve the data for this executor
        data = ExecutorData.get(cons);

        try {
            // Since we want to produce many kernels on the same device, we pick the device based on some kernel
            device = Cashmere.getDevice("grayscaleKernel");

            if (node1 != node2) {
                retrieveRemoteFrequencyDomain(cons, executor);

                computeCorrelations(cons);
                logger.debug("Leaf: finishing");
                return FINISH;

            } else {
                // In this case, all time domain patterns must be available in the cache.
                if (indicesI[0] == indicesJ[0]) {
                    assert (indicesI.length == indicesJ.length);
                    // we need both the regular and flipped frequency domains as the i and j range overlap
                    retrieveFrequencyDomain(cons, node1, indicesI, filesI, false, true, device);
                } else {
                    retrieveFrequencyDomain(cons, node1, indicesI, filesI, REGULAR, false, device);
                    retrieveFrequencyDomain(cons, node2, indicesJ, filesJ, FLIPPED, false, device);
                }

                computeCorrelations(cons);
                logger.debug("Leaf: finishing");
                return FINISH;
            }

        } catch (IOException | CashmereNotAvailable | LibFuncNotAvailable | NoSuitableExecutorException e) {
            logger.error("Got exception", e);
            throw new Error(e);
        }
    }

    @Override
    public int process(Constellation cons, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation cons) {
        if (logger.isDebugEnabled()) {
            synchronized (LeafCorrelationsActivity.class) {
                inFlight--;
                logger.debug("Leaf: cleanup; {} in flight", inFlight);
            }
        }
        timer.stop(event);
        // notify the progress
        cons.send(new Event(identifier(), progressActivityID, correlations.size()));
        // send the result back to the parent
        cons.send(new Event(identifier(), parent, correlations));
    }

    // private methods

    // cooperatively retrieve the frequency domain of a specific range
    private void retrieveFrequencyDomain(Constellation cons, int node, int[] indices, File[] files, boolean flipped, boolean both,
            Device device) throws CashmereNotAvailable, IOException, LibFuncNotAvailable {

        /*
         * We keep track of which in the range has already be done, since another thread may be processing it.  This means that
         * executors are cooperatively preparing the data.
         */

        boolean[] done = new boolean[indices.length];
        do {
            for (int i = 0; i < indices.length; i++) {
                if (!done[i]) {
                    retrieveFrequencyDomain(cons, node, i, indices, files, flipped, both, device, done);
                }
            }
        } while (!allDone(done));
    }

    private static boolean allDone(boolean[] done) {
        for (boolean b : done) {
            if (!b) {
                return false;
            }
        }
        return true;
    }

    /* pre:
     * - the current thread does not hold locks for index
     * post:
     * - done[i] = true
     *   - both = true
     *     - the current thread holds a read lock for index flipped
     *     - the current thread holds a read lock for index regular
     *     - noisePatternsXFreq[i] is set
     *     - noisePatternsYFreq[i] is set
     *   - both = false
     *     - the current thread holds a read lock for index flipped
     *     - noisePatternsX/YFreq[i] is set
     * - done[i] = false
     *   - the current thread does not hold locks for index
     */
    private void retrieveFrequencyDomain(Constellation cons, int node, int i, int[] indices, File[] files, boolean flipped,
            boolean both, Device device, boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        if (!both) {
            retrieveFrequencyDomainSingle(cons, i, flipped, indices, files, device, done);
        } else {
            retrieveFrequencyDomainBoth(cons, i, indices, files, device, done);
        }

    }

    /**
     * Retrieve frequency domains, in the case that the images are owned by two hosts.
     *
     * @param cons
     * @throws NoSuitableExecutorException
     * @throws LibFuncNotAvailable
     * @throws CashmereNotAvailable
     */
    private void retrieveRemoteFrequencyDomain(Constellation cons, String executor)
            throws NoSuitableExecutorException, CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int[] indices = node1 == NodeInformation.ID ? indicesJ : indicesI;
        boolean flipped = node1 == NodeInformation.ID;
        File[] files = node1 == NodeInformation.ID ? filesJ : filesI;
        int node = node1 == NodeInformation.ID ? node2 : node1;

        int[] toRequest = new int[indices.length];
        File[] reqFiles = new File[indices.length];
        int nRequests = 0;
        int[] haveTimePatterns = new int[indices.length];
        int nHaveTimePatterns = 0;
        ArrayList<LockToken<Pointer>> ptrlocks = new ArrayList<LockToken<Pointer>>();
        ArrayList<LockToken<Buffer>> buflocks = new ArrayList<LockToken<Buffer>>();

        needed = new boolean[indices.length];

        for (int i = 0; i < indices.length; i++) {
            buflocks.add(null);
            ptrlocks.add(null);
        }

        // First try and get locks on all freq domain noise patterns.
        boolean done = false;
        while (!done) {
            done = true;
            for (int i = 0; i < indices.length; i++) {
                int index = indices[i];
                LockToken<Pointer> lt = ptrlocks.get(i);
                if (lt == null) {
                    try {
                        lt = NoisePatternCache.lockNoisePatternFreq(index, flipped);
                        if (lt.readLock()) {
                            if (logger.isDebugEnabled()) {
                                logger.debug("noise pattern freq {} {} on the device", index, string(flipped));
                            }
                            setNoisePatternFreq(i, flipped, lt.availableElement);
                            // Keep the lock ...
                        }
                        ptrlocks.set(i, lt);
                    } catch (LockException e) {
                        done = false;
                    }
                }
            }
        }

        // Now we have read locks or write locks.
        // Readlocks: fine. WriteLocks: depends: do we have the time domain pattern?
        done = false;
        while (!done) {
            done = true;
            for (int i = 0; i < indices.length; i++) {
                int index = indices[i];
                LockToken<Buffer> memlt = buflocks.get(i);
                if (memlt == null) {
                    LockToken<Pointer> lt = ptrlocks.get(i);
                    if (lt.writeLock()) {
                        try {
                            memlt = NoisePatternCache.lockNoisePattern(index);
                            buflocks.set(i, memlt);
                            if (memlt.readLock()) {
                                // Yes, we have it!
                                haveTimePatterns[nHaveTimePatterns++] = i;
                            } else {
                                reqFiles[nRequests] = files[i];
                                toRequest[nRequests++] = index;
                                needed[i] = true;
                            }
                        } catch (LockException e) {
                            done = false;
                        }
                    }
                }
            }
        }

        if (nRequests != 0) {
            toRequest = Arrays.copyOf(toRequest, nRequests);
            logger.debug(identifier().toString() + ": requesting NPs " + Arrays.toString(toRequest) + " from node " + node
                    + ", flipped = " + flipped);
            cons.submit(
                    new FetchPatternActivity(reqFiles, toRequest, this, providers[node][random.nextInt(providers[node].length)]));
        }

        // Provide freq patterns that we can now. Note that we do this in the latency of the request if there is one.
        for (int j = 0; j < nHaveTimePatterns; j++) {
            int i = haveTimePatterns[j];
            int index = indices[i];
            LockToken<Buffer> memlt = buflocks.get(i);
            buflocks.set(i, null);
            assert (memlt.readLock());
            copyNoisePatternToDevice(index, device, memlt.availableElement);
            NoisePatternCache.unlockNoisePattern(index);
            LockToken<Pointer> lt = ptrlocks.get(i);
            assert (lt.writeLock());
            ptrlocks.set(i, null);
            produceNoisePatternFreq(cons, i, index, flipped, device, lt.availableElement);
            NoisePatternCache.toReadLockNoisePatternFreq(index, flipped);
        }

        // Also retrieve the frequency domain patterns for "our" images. We also do this in the latency in case we have done a
        // remote request.
        if (flipped) {
            retrieveFrequencyDomain(cons, node1, indicesI, filesI, REGULAR, false, device);
        } else {
            retrieveFrequencyDomain(cons, node2, indicesJ, filesJ, FLIPPED, false, device);
        }

        if (nRequests != 0) {
            Timer waitTimer = Cashmere.getTimer("java", executor, "waiting for noise patterns");
            int ev = waitTimer.start();
            synchronized (this) {
                while (message == null) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        // ignore
                    }
                }
            }
            waitTimer.stop(ev);
            NPBuffer data = (NPBuffer) message.getData();
            int count = 0;
            // Note that we have kept the freq writelocks that we still need.
            // Now, for all freq domain write locks compute freq domain pattern.
            for (int i = 0; i < indices.length; i++) {
                int index = indices[i];
                if (needed[i]) {
                    if (index != data.indices[count]) {
                        throw new Error("Internal error");
                    }
                    Buffer buf = data.buf[count++];
                    LockToken<Pointer> ltp = ptrlocks.get(i);
                    if (ltp.writeLock()) {
                        LockToken<Buffer> ltb = buflocks.get(i);
                        if (ltb.writeLock()) {
                            ltb.availableElement.getByteBuffer().clear();
                            buf.getByteBuffer().rewind();
                            ltb.availableElement.getByteBuffer().put(buf.getByteBuffer()); // pity that we have to copy
                            ltb.availableElement.getByteBuffer().rewind();
                        } else {
                            assert (ltb.readLock());
                        }
                        copyNoisePatternToDevice(index, device, ltb.availableElement);
                        NoisePatternCache.unlockNoisePattern(index);
                        try {
                            produceNoisePatternFreq(cons, i, index, flipped, device, ltp.availableElement);
                        } catch (CashmereNotAvailable | LibFuncNotAvailable e) {
                            logger.error("Got exception", e);
                            throw new Error(e);
                        }
                        NoisePatternCache.toReadLockNoisePatternFreq(index, flipped);
                    } else {
                        throw new Error("Internal error");
                    }
                    ByteBufferCache.makeAvailableByteBuffer(buf.getByteBuffer());
                }
            }
        }
    }

    /*
     * Retrieves a frequency domain of a noise pattern indicated by index and flipped.  We try to lock the noise pattern and if
     * this succeeds then we may have read lock or write lock.  If we hold a read lock, then we are done, if we hold a write lock
     * we are supposed to produce the noise pattern.  Afterwards, we have to move the write lock to a read lock.
     *
     * This version assumes that the time domain noise patterns are in the cache.
     *
     * pre:
     * - done[i] = false
     * - the current thread does not hold a lock for index
     * post:
     * - done[i] = true
     * - we hold a readlock of index flipped
     * - noisePatternsX/YFreq[i] is set
     */
    private void retrieveFrequencyDomainSingle(Constellation cons, int i, boolean flipped, int[] indices, File[] files,
            Device device, boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int index = indices[i];

        LockToken<Pointer> lt = null;
        try {
            lt = NoisePatternCache.lockNoisePatternFreq(index, flipped);

            if (lt.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} {} on the device", i, index, string(flipped));
                }
                setNoisePatternFreq(i, flipped, lt.availableElement);
                done[i] = true;
            } else if (lt.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} {} not on the device", i, index, string(flipped));
                }

                /*
                 * we are going to try to get the time domain noise pattern to the device.  If that succeeds, we produce the
                 * frequency domain noise pattern, otherwise, we did not succeed and we have to register that this noise pattern
                 * is not kept in the cache since it does not contain useful information.
                 */
                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, flipped, device, lt.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(index, flipped);
                    done[i] = true;
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, flipped);
                    done[i] = false;
                }
            } else {
                throw new Error("should not happen");
            }
        } catch (LockException e) {
            // ignore
        }
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
    private boolean noisePatternOnDevice(Constellation cons, int index, File file, Device device)
            throws CashmereNotAvailable, LibFuncNotAvailable, IOException {
        try {
            LockToken<Buffer> lt = NoisePatternCache.lockNoisePattern(index);
            if (lt.readLock()) {
                copyNoisePatternToDevice(index, device, lt.availableElement);
                NoisePatternCache.unlockNoisePattern(index);
                return true;
            } else if (lt.writeLock()) {
                produceNoisePattern(cons, index, file, device, lt.availableElement);
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
     * - the current thread holds a write lock for noise pattern index
     * post:
     * - noise pattern index is on device
     * - noise pattern index is in the cache
    */
    private void produceNoisePattern(Constellation cons, int index, File file, Device device, Buffer noisePattern)
            throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        if (logger.isDebugEnabled()) {
            logger.debug("computing noise pattern {}, putting it on the device", index);
        }

        ComputeNoisePattern.computePRNU_MC(index, file, h, w, cons.identifier().toString(), device, data);
        // get the data from the device, noisePattern points to memory in the cache
        device.get(noisePattern, data.noisePattern);
    }

    /* post:
     * - noisePatternsXFreq[i] or noisePatternsYFreq[i] is set
     */
    void setNoisePatternFreq(int i, boolean flipped, Pointer noisePatternFreq) {
        if (flipped) {
            noisePatternsYFreq[i] = noisePatternFreq;
        } else {
            noisePatternsXFreq[i] = noisePatternFreq;
        }
    }

    /*
     * pre:
     * - the current thread holds a read lock for noisePattern
     * post:
     * - the noisePattern is on the device
     */
    private void copyNoisePatternToDevice(int index, Device device, Buffer noisePattern) {
        if (logger.isDebugEnabled()) {
            logger.debug("copying noise pattern {} to the device", index);
        }

        device.copy(noisePattern, data.noisePattern);
    }

    /*
     * pre:
     * - the current thread holds a write lock for noise pattern freq index
     * flipped
     * post:
     * - the device is associated with the index
     * - noisePatternsX/YFreq[i] is set
     */
    private void produceNoisePatternFreq(Constellation cons, int i, int index, boolean flipped, Device device,
            Pointer noisePatternFreq) throws CashmereNotAvailable, LibFuncNotAvailable {

        if (logger.isDebugEnabled()) {
            logger.debug("Computing frequency domain of {} {}, putting it on the device", index, string(flipped));
        }

        NoisePatternCache.setDevice(index, device);
        String executor = cons.identifier().toString();

        ComputeFrequencyDomain.computeFreq(device, noisePatternFreq, h, w, flipped, executor, data);

        setNoisePatternFreq(i, flipped, noisePatternFreq);
    }

    /*
     * Retrieve the frequency domain noise patterns for a noise pattern indicated by index, for both regular and flipped.  This
     * method deals with five cases:
     * - read lock for both flipped and regular
     * - read lock for flipped, write lock for regular
     * - write lock for flipped, read lock for regular
     * - write lock for both flipped and regular
     * - failing to get a lock for one of regular, flipped, in which' case we fail
     *
     * pre:
     * - done[i] = false
     * - the current thread does not hold locks for index
     * post:
     * - done[i] = true
     * - we hold a readlock of index flipped
     * - we hold a readlock of index regular
     * - noisePatternsXFreq[i] is set
     * - noisePatternsYFreq[i] is set
     */
    private void retrieveFrequencyDomainBoth(Constellation cons, int i, int[] indices, File[] files, Device device,
            boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int index = indices[i];

        LockToken<Pointer> ltRegular = null;
        LockToken<Pointer> ltFlipped = null;
        try {
            ltRegular = NoisePatternCache.lockNoisePatternFreq(index, REGULAR);
            ltFlipped = NoisePatternCache.lockNoisePatternFreq(index, FLIPPED);
            done[i] = true;

            if (ltRegular.readLock() && ltFlipped.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("both noise patterns freq {} {} flipped and regular on the device", i, index);
                }

                setNoisePatternFreq(i, REGULAR, ltRegular.availableElement);
                setNoisePatternFreq(i, FLIPPED, ltFlipped.availableElement);
            } else if (ltRegular.readLock() && ltFlipped.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} regular on the device", i, index);
                }

                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, FLIPPED, device, ltFlipped.availableElement);
                    setNoisePatternFreq(i, REGULAR, ltRegular.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(index, FLIPPED);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, FLIPPED);
                    done[i] = false;
                }

            } else if (ltRegular.writeLock() && ltFlipped.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} flipped on the device", i, index);
                }

                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, REGULAR, device, ltRegular.availableElement);
                    setNoisePatternFreq(i, FLIPPED, ltFlipped.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreq(index, FLIPPED);
                    done[i] = false;
                }

            } else if (ltRegular.writeLock() && ltFlipped.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("none of noise patterns freq {} {} on the device", i, index);
                }

                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, FLIPPED, device, ltFlipped.availableElement);
                    produceNoisePatternFreq(cons, i, index, REGULAR, device, ltRegular.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(index, FLIPPED);
                    NoisePatternCache.toReadLockNoisePatternFreq(index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, FLIPPED);
                    done[i] = false;
                }
            } else {
                throw new Error("should not happen");
            }
        } catch (LockException e) {
            if (ltRegular != null) {
                if (ltRegular.writeLock()) {
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, REGULAR);
                }
            }
            if (ltFlipped != null) {
                if (ltFlipped.writeLock()) {
                    NoisePatternCache.unlockNoisePatternFreqRemove(index, FLIPPED);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, FLIPPED);
                }
            }
            done[i] = false;
        }
    }

    // correlations

    /*
     * Compute the correlations for the ranges.  We hold read locks for startI-endI regular and for startJ-endJ flipped.  We
     * unlock the read lock when done.
     */
    private void computeCorrelations(Constellation cons) throws CashmereNotAvailable, LibFuncNotAvailable {

        if (indicesI[0] == indicesJ[0]) {
            for (int i = 0; i < indicesI.length; i++) {
                NoisePatternCache.unlockNoisePatternFreq(indicesI[i], true);
                for (int j = i + 1; j < indicesJ.length; j++) {
                    computeCorrelation(cons, i, j);
                }
                NoisePatternCache.unlockNoisePatternFreq(indicesI[i], false);
            }
        } else {
            for (int i = 0; i < indicesI.length; i++) {
                for (int j = 0; j < indicesJ.length; j++) {
                    computeCorrelation(cons, i, j);
                }
                NoisePatternCache.unlockNoisePatternFreq(indicesI[i], false);
            }
            for (int j = 0; j < indicesJ.length; j++) {
                NoisePatternCache.unlockNoisePatternFreq(indicesJ[j], true);
            }
        }
    }

    private void computeCorrelation(Constellation cons, int i, int j) throws CashmereNotAvailable, LibFuncNotAvailable {

        String executor = cons.identifier().toString();

        Correlation c = new Correlation(indicesI[i], indicesJ[j]);

        Pointer x = noisePatternsXFreq[i];
        Pointer y = noisePatternsYFreq[j];

        c.coefficient = ComputeCorrelation.correlateMC(indicesI[i], indicesJ[j], x, y, h, w, executor, device, data);
        correlations.add(c);

        if (logger.isDebugEnabled()) {
            int i1 = indicesI[i] < indicesJ[j] ? indicesI[i] : indicesJ[j];
            int i2 = indicesI[i] >= indicesJ[j] ? indicesI[i] : indicesJ[j];
            logger.debug("Correlation of {},{} is {}", i1, i2, c.coefficient);
        }

    }

    private String string(boolean flipped) {
        return flipped ? "flipped" : "regular";
    }

    public synchronized void pushMessage(Event event2) {
        this.message = event2;
        notifyAll();
    }
}
