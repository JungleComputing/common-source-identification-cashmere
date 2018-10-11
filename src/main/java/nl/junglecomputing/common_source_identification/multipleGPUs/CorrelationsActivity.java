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
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.OrContext;
import ibis.constellation.Timer;
import ibis.constellation.util.ByteBufferCache;
import nl.junglecomputing.common_source_identification.cpu.Correlation;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;
import nl.junglecomputing.common_source_identification.dedicated_activities.Correlations;
import nl.junglecomputing.common_source_identification.device_mem_cache.LockToken;
import nl.junglecomputing.common_source_identification.main_mem_cache.LockException;
import nl.junglecomputing.common_source_identification.mc.ComputeCorrelation;
import nl.junglecomputing.common_source_identification.mc.ComputeFrequencyDomain;
import nl.junglecomputing.common_source_identification.mc.ComputeNoisePattern;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;

class CorrelationsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationsActivity");

    // Constellation logic
    public static final String LABEL = "correlation";
    final protected ActivityIdentifier parent;

    // logic for subdivision of correlations
    protected int node1;
    protected int node2;

    protected int[] indicesI;
    protected int[] indicesJ;

    protected File[] filesI;
    protected File[] filesJ;

    protected int level;

    // Correlation logic
    protected int h;
    protected int w;

    protected transient Correlations correlations;
    protected transient int nrReceivedCorrelations;
    protected transient int nrCorrelationsToReceive;

    // identifies the activity that we notify the amount of correlations we have processed
    final protected ActivityIdentifier progressActivityID;

    final protected ActivityIdentifier[][] providers;

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

    private transient DeviceInfo deviceInfo;
    private transient Device device;
    private transient int deviceNo;

    private transient Event messageI = null;
    private transient Event messageJ = null;

    private transient Random random;

    private transient FetchPatternActivity fI;

    private transient FetchPatternActivity fJ;

    private transient String executor;

    private transient boolean leaf;

    CorrelationsActivity(ActivityIdentifier parent, ActivityIdentifier progressActivityID, int[] indicesI, int[] indicesJ,
            int node1, int node2, File[] filesI, File[] filesJ, int h, int w, int level, ActivityIdentifier[][] providers) {

        super(node1 == node2 ? new OrContext(new Context(LABEL + node1, level), new Context(LABEL, level))
                : new OrContext(new Context(LABEL + node1, level), new Context(LABEL + node2, level), new Context(LABEL, level)),
                true);

        logger.debug("Creating Correlation with context {}, node1 = {}, node2 = {}, size1 = {}, size2 = {}",
                getContext().toString(), node1, node2, indicesI.length, indicesJ.length);

        this.parent = parent;
        this.progressActivityID = progressActivityID;
        this.node1 = node1;
        this.node2 = node2;
        this.providers = providers;

        this.filesI = filesI;
        this.filesJ = filesJ;

        this.indicesI = indicesI;
        this.indicesJ = indicesJ;

        this.level = level;

        this.h = h;
        this.w = w;
    }

    @Override
    public int initialize(Constellation cons) {

        this.correlations = new Correlations();
        this.nrReceivedCorrelations = 0;
        this.nrCorrelationsToReceive = 0;

        if (logger.isDebugEnabled()) {
            logger.debug("Running Correlation, node1 = {}, node2 = {}, size1 = {}, size2 = {}", node1, node2, indicesI.length,
                    indicesJ.length);
            synchronized (CorrelationsActivity.class) {
                inFlight++;
                logger.debug("{} in flight", inFlight);
            }
        }

        executor = cons.identifier().toString();

        // thresholdDC contains the total number of images a tile may contain.
        deviceInfo = DeviceInfo.getDeviceInfo(executor, LABEL);
        int threshold = deviceInfo.getThreshold() / 2;
        if (indicesI.length <= threshold && indicesJ.length <= threshold) {
            leaf = true;
            return initializeLeaf(cons);
        }
        return initializeTree(cons);
    }

    @Override
    public void cleanup(Constellation cons) {
        if (logger.isDebugEnabled()) {
            synchronized (CorrelationsActivity.class) {
                inFlight--;
                logger.debug("{} in flight", inFlight);
            }
        }
        if (leaf) {
            timer.stop(event);
            // notify the progress
            cons.send(new Event(identifier(), progressActivityID, correlations.size()));
        }
        cons.send(new Event(identifier(), parent, correlations));
    }

    @Override
    public int process(Constellation cons, Event event) {
        if (!leaf) {
            Object data = event.getData();

            if (data instanceof Correlations) {
                Correlations correlations = (Correlations) data;
                return processCorrelations(correlations);
            } else {
                throw new Error("Unknown type of data");
            }
        } else {
            return FINISH;
        }
    }

    // private methods

    private int initializeLeaf(Constellation cons) {

        this.noisePatternsXFreq = new Pointer[indicesI.length];
        this.noisePatternsYFreq = new Pointer[indicesJ.length];

        random = new Random();

        this.noisePatternsXFreq = new Pointer[indicesI.length];
        this.noisePatternsYFreq = new Pointer[indicesJ.length];

        random = new Random();

        this.timer = Cashmere.getTimer("java", executor, "leaf correlations");
        this.event = timer.start();

        try {
            // Since we want to produce many kernels on the same device, we pick the device based on some kernel
            device = deviceInfo.getDevice();
            data = deviceInfo.getExecutorData();
            deviceNo = deviceInfo.getDeviceNo();

            if (node1 != node2 || node1 != NodeInformation.ID) {
                retrieveRemoteFrequencyDomain(cons);
            } else {
                // In this case, all time domain patterns must be available in the cache.
                if (indicesI[0] == indicesJ[0]) {
                    assert (indicesI.length == indicesJ.length);
                    // we need both the regular and flipped frequency domains as the i and j range overlap
                    retrieveFrequencyDomain(cons, indicesI, filesI, false, true);
                } else {
                    retrieveFrequencyDomain(cons, indicesI, filesI, REGULAR, false);
                    retrieveFrequencyDomain(cons, indicesJ, filesJ, FLIPPED, false);
                }
            }

            computeCorrelations(cons);
            logger.debug("Leaf: finishing");
            return FINISH;

        } catch (IOException | CashmereNotAvailable | LibFuncNotAvailable | NoSuitableExecutorException e) {
            logger.error("Got exception", e);
            throw new Error(e);
        }
    }

    private int initializeTree(Constellation cons) {

        /* The following code splits up the range in the i direction and the range in the j direction based on the threshold.
         * There is a nested for loop that will submit the nested CorrelationsActiviities.
         */

        // determining the number of activities for i and j
        int nrActivitiesI = indicesI.length;
        int nrActivitiesJ = indicesJ.length;

        // The following code determines the number of iterations of the nested for-loop and the number of activities each
        // subdivision will contain
        int nrIterationsI;
        int nrIterationsJ;

        int nrActivitiesPerIterI;
        int nrActivitiesPerIterJ;

        int threshold = deviceInfo.getThreshold() / 2;

        // The case where we can subdivide both ranges in two.
        nrIterationsI = nrActivitiesI > threshold ? 2 : 1;
        nrIterationsJ = nrActivitiesJ > threshold ? 2 : 1;

        nrActivitiesPerIterI = getNrActivitiesPerIter(nrActivitiesI, nrIterationsI);
        nrActivitiesPerIterJ = getNrActivitiesPerIter(nrActivitiesJ, nrIterationsJ);

        // The following nested loop determines the start and end of the ranges and submits new CorrelationsActivities.
        for (int i = 0; i < nrIterationsI; i++) {
            int startIndexI = i * nrActivitiesPerIterI;
            int endIndexI = Math.min(startIndexI + nrActivitiesPerIterI, indicesI.length);

            for (int j = 0; j < nrIterationsJ; j++) {
                int startIndexJ = j * nrActivitiesPerIterJ;
                int endIndexJ = Math.min(startIndexJ + nrActivitiesPerIterJ, indicesJ.length);

                if (node1 != node2 || indicesI[0] != indicesJ[0] || startIndexI <= startIndexJ) {
                    // Ranges are different, or else do only triangle.
                    try {
                        submitActivity(cons, Arrays.copyOfRange(indicesI, startIndexI, endIndexI),
                                Arrays.copyOfRange(indicesJ, startIndexJ, endIndexJ),
                                Arrays.copyOfRange(filesI, startIndexI, endIndexI),
                                Arrays.copyOfRange(filesJ, startIndexJ, endIndexJ));
                    } catch (NoSuitableExecutorException e) {
                        logger.error("Could not submit activity", e);
                        return FINISH;
                    }
                    nrCorrelationsToReceive++;
                }
            }
        }
        return SUSPEND;
    }

    private int getNrActivitiesPerIter(int nrActivities, int nrIterations) {
        return getDivide(nrActivities, nrIterations);
    }

    private int getNrIterations(int nrActivities, int threshold) {
        return getDivide(nrActivities, threshold);
    }

    private int getDivide(int numerator, int denominator) {
        int rest = numerator % denominator;
        int division = numerator / denominator;
        if (rest == 0) {
            return division;
        } else {
            return division + 1;
        }
    }

    private void submitActivity(Constellation cons, int[] indicesI, int[] indicesJ, File[] filesI, File[] filesJ)
            throws NoSuitableExecutorException {

        CorrelationsActivity activity = new CorrelationsActivity(identifier(), progressActivityID, indicesI, indicesJ, node1,
                node2, filesI, filesJ, h, w, node1 == node2 ? 100 : level + 1, providers);
        cons.submit(activity);
    }

    private int processCorrelations(Correlations correlations) {
        this.correlations.addAll(correlations);
        nrReceivedCorrelations++;
        if (logger.isDebugEnabled()) {
            logger.debug("Received correlations {}/{}", nrReceivedCorrelations, nrCorrelationsToReceive);
        }
        if (nrReceivedCorrelations == nrCorrelationsToReceive) {
            return FINISH;
        } else {
            return SUSPEND;
        }
    }

    // cooperatively retrieve the frequency domain of a specific range
    private void retrieveFrequencyDomain(Constellation cons, int[] indices, File[] files, boolean flipped, boolean both)
            throws CashmereNotAvailable, IOException, LibFuncNotAvailable {

        /*
         * We keep track of which in the range has already be done, since another thread may be processing it.  This means that
         * executors are cooperatively preparing the data.
         */

        boolean[] done = new boolean[indices.length];
        do {
            for (int i = 0; i < indices.length; i++) {
                if (!done[i]) {
                    retrieveFrequencyDomain(cons, i, indices, files, flipped, both, done);
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
    private void retrieveFrequencyDomain(Constellation cons, int i, int[] indices, File[] files, boolean flipped, boolean both,
            boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        if (!both) {
            retrieveFrequencyDomainSingle(cons, i, flipped, indices, files, done);
        } else {
            retrieveFrequencyDomainBoth(cons, i, indices, files, done);
        }
    }

    private FetchPatternActivity getFetcher(int node, int[] indices, File[] files, ArrayList<LockToken<Pointer>> ptrlocks,
            ArrayList<LockToken<Buffer>> buflocks, ArrayList<LockToken<Pointer>> ptrlocksFlipped, boolean flipped, boolean both) {

        if (node != NodeInformation.ID) {
            Timer timer = Cashmere.getTimer("java", executor, "getFetcher");
            int ev = timer.start();
            for (int i = 0; i < indices.length; i++) {
                buflocks.add(null);
                ptrlocks.add(null);
                ptrlocksFlipped.add(null);
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
                            lt = NoisePatternCache.lockNoisePatternFreq(deviceNo, index, flipped);
                            if (logger.isDebugEnabled()) {
                                logger.debug("getFetcher: locking freq {} {} {}", index, string(flipped),
                                        lt.readLock() ? "read" : "write");
                            }
                            if (lt.readLock()) {
                                setNoisePatternFreq(i, flipped, lt.availableElement);
                                // Keep the lock ...
                            }
                            ptrlocks.set(i, lt);
                        } catch (LockException e) {
                            done = false;
                        }
                    }
                    if (both) {
                        lt = ptrlocksFlipped.get(i);
                        if (lt == null) {
                            try {
                                lt = NoisePatternCache.lockNoisePatternFreq(deviceNo, index, FLIPPED);
                                if (logger.isDebugEnabled()) {
                                    logger.debug("getFetcher: locking freq {} {} {}", index, string(FLIPPED),
                                            lt.readLock() ? "read" : "write");
                                }
                                if (lt.readLock()) {
                                    setNoisePatternFreq(i, FLIPPED, lt.availableElement);
                                    // Keep the lock ...
                                }
                                ptrlocksFlipped.set(i, lt);
                            } catch (LockException e) {
                                done = false;
                            }
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
                        LockToken<Pointer> flt = ptrlocksFlipped.get(i);
                        if (lt.writeLock() || (flt != null && flt.writeLock())) {
                            try {
                                memlt = NoisePatternCache.lockNoisePattern(index);
                                if (logger.isDebugEnabled()) {
                                    logger.debug("getFetcher: locking time {} {}", index, memlt.readLock() ? "read" : "write");
                                }
                                buflocks.set(i, memlt);
                            } catch (LockException e) {
                                done = false;
                            }
                        }
                    }
                }
            }

            int[] toRequest = new int[indices.length];
            File[] requestFiles = new File[indices.length];
            int count = 0;
            for (int i = 0; i < indices.length; i++) {
                LockToken<Buffer> memlt = buflocks.get(i);
                if (memlt != null && memlt.writeLock()) {
                    toRequest[count] = indices[i];
                    requestFiles[count] = files[i];
                    count++;
                }
            }
            toRequest = Arrays.copyOf(toRequest, count);
            requestFiles = Arrays.copyOf(requestFiles, count);
            timer.stop(ev);
            return new FetchPatternActivity(requestFiles, toRequest, this,
                    providers[node][random.nextInt(providers[node].length)]);
        }
        return null;
    }

    void handleAlreadyPresent(Constellation constellation, int[] indices, ArrayList<LockToken<Pointer>> ptrlocks,
            ArrayList<LockToken<Pointer>> ptrlocksFlipped, ArrayList<LockToken<Buffer>> buflocks, boolean flipped, boolean both)
            throws CashmereNotAvailable, LibFuncNotAvailable {
        for (int i = 0; i < indices.length; i++) {
            LockToken<Buffer> memlt = buflocks.get(i);
            if (memlt != null && memlt.readLock()) {
                int index = indices[i];
                buflocks.set(i, null);
                copyNoisePatternToDevice(index, device, memlt.availableElement);
                NoisePatternCache.unlockNoisePattern(index);
                if (logger.isDebugEnabled()) {
                    logger.debug("handleAlreadyPresent: unlocking time {}", index);
                }
                LockToken<Pointer> lt = ptrlocks.get(i);
                ptrlocks.set(i, null);
                if (lt != null && lt.writeLock()) {
                    produceNoisePatternFreq(constellation, i, index, flipped, device, lt.availableElement);
                    if (logger.isDebugEnabled()) {
                        logger.debug("handleAlreadyPresent: toreadlock freq {} {}", index, string(flipped));
                    }
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, flipped);
                }
                lt = ptrlocksFlipped.get(i);
                ptrlocksFlipped.set(i, null);
                if (lt != null && lt.writeLock()) {
                    produceNoisePatternFreq(constellation, i, index, FLIPPED, device, lt.availableElement);
                    if (logger.isDebugEnabled()) {
                        logger.debug("handleAlreadyPresent: toreadlock freq {} {}", index, string(FLIPPED));
                    }
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, FLIPPED);
                }

            }
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
    private void retrieveRemoteFrequencyDomain(Constellation cons)
            throws NoSuitableExecutorException, CashmereNotAvailable, LibFuncNotAvailable, IOException {

        ArrayList<LockToken<Pointer>> ptrlocksI = new ArrayList<LockToken<Pointer>>();
        ArrayList<LockToken<Buffer>> buflocksI = new ArrayList<LockToken<Buffer>>();
        ArrayList<LockToken<Pointer>> ptrlocksJ = new ArrayList<LockToken<Pointer>>();
        ArrayList<LockToken<Buffer>> buflocksJ = new ArrayList<LockToken<Buffer>>();
        ArrayList<LockToken<Pointer>> ptrlocksFlipped = new ArrayList<LockToken<Pointer>>();
        ArrayList<LockToken<Pointer>> dummy = new ArrayList<LockToken<Pointer>>();
        boolean both = node1 == node2 && indicesI[0] == indicesJ[0];
        fI = getFetcher(node1, indicesI, filesI, ptrlocksI, buflocksI, ptrlocksFlipped, REGULAR, both);
        fJ = null;
        if (!both) {
            fJ = getFetcher(node2, indicesJ, filesJ, ptrlocksJ, buflocksJ, dummy, FLIPPED, false);
        }

        if (fI != null) {
            logger.debug(identifier().toString() + ": requesting " + Arrays.toString(fI.request.indices) + " from node " + node1);
            cons.submit(fI);
        }
        if (fJ != null) {
            logger.debug(identifier().toString() + ": requesting " + Arrays.toString(fJ.request.indices) + " from node " + node2);
            cons.submit(fJ);
        }

        // Retrieve freq domains from what we already have.
        if (fI == null) {
            retrieveFrequencyDomain(cons, indicesI, filesI, REGULAR, false);
        } else {
            handleAlreadyPresent(cons, indicesI, ptrlocksI, ptrlocksFlipped, buflocksI, REGULAR, both);
        }

        if (!both) {
            if (fJ == null) {
                retrieveFrequencyDomain(cons, indicesJ, filesJ, FLIPPED, false);
            } else {
                handleAlreadyPresent(cons, indicesJ, ptrlocksJ, dummy, buflocksJ, FLIPPED, false);
            }
        }

        Timer waitTimer = Cashmere.getTimer("java", executor, "waiting for noise patterns");
        if (fI != null || fJ != null) {
            int ev = waitTimer.start();
            synchronized (this) {
                while (messageI == null && messageJ == null) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        // ignore
                    }
                }
            }
            waitTimer.stop(ev);
            if (messageI != null) {
                handleMessageData(cons, (NPBuffer) messageI.getData(), indicesI, ptrlocksI, ptrlocksFlipped, buflocksI, REGULAR);
                messageI = null;
            } else {
                handleMessageData(cons, (NPBuffer) messageJ.getData(), indicesJ, ptrlocksJ, dummy, buflocksJ, FLIPPED);
                messageJ = null;
            }
        }
        if (fI != null && fJ != null) {
            int ev = waitTimer.start();
            synchronized (this) {
                while (messageI == null && messageJ == null) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        // ignore
                    }
                }
            }
            waitTimer.stop(ev);
            if (messageI != null) {
                handleMessageData(cons, (NPBuffer) messageI.getData(), indicesI, ptrlocksI, ptrlocksFlipped, buflocksI, REGULAR);
                messageI = null;
            } else {
                handleMessageData(cons, (NPBuffer) messageJ.getData(), indicesJ, ptrlocksJ, dummy, buflocksJ, FLIPPED);
                messageJ = null;
            }
        }
    }

    private void handleMessageData(Constellation constellation, NPBuffer data, int[] indices,
            ArrayList<LockToken<Pointer>> ptrlocks, ArrayList<LockToken<Pointer>> ptrlocksFlipped,
            ArrayList<LockToken<Buffer>> buflocks, boolean flipped) throws CashmereNotAvailable, LibFuncNotAvailable {
        int count = 0;
        // Note that we have kept the freq writelocks that we still need.
        // Now, for all freq domain write locks compute freq domain pattern.
        Timer timer = Cashmere.getTimer("java", executor, "process message");
        int ev = timer.start();
        for (int i = 0; i < indices.length; i++) {
            int index = indices[i];
            LockToken<Buffer> ltb = buflocks.get(i);
            if (ltb != null && ltb.writeLock()) {
                if (index != data.indices[count]) {
                    throw new Error("Internal error");
                }
                Buffer buf = data.buf[count++];
                ltb.availableElement.getByteBuffer().clear();
                buf.getByteBuffer().rewind();
                ltb.availableElement.getByteBuffer().put(buf.getByteBuffer()); // pity that we have to copy
                ltb.availableElement.getByteBuffer().rewind();
                ByteBufferCache.makeAvailableByteBuffer(buf.getByteBuffer());
                copyNoisePatternToDevice(index, device, ltb.availableElement);
                NoisePatternCache.unlockNoisePattern(index);
                if (logger.isDebugEnabled()) {
                    logger.debug("handleMessageData: unlock time {}", index);
                }
                LockToken<Pointer> ltp = ptrlocks.get(i);
                if (ltp != null && ltp.writeLock()) {
                    produceNoisePatternFreq(constellation, i, index, flipped, device, ltp.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, flipped);
                    if (logger.isDebugEnabled()) {
                        logger.debug("handleMessageData: toreadlock freq {} {}", index, string(flipped));
                    }
                }
                ltp = ptrlocksFlipped.get(i);
                if (ltp != null && ltp.writeLock()) {
                    produceNoisePatternFreq(constellation, i, index, FLIPPED, device, ltp.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, FLIPPED);
                    if (logger.isDebugEnabled()) {
                        logger.debug("handleMessageData: toreadlock freq {} {}", index, string(FLIPPED));
                    }
                }
            }
        }
        timer.stop(ev);
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
            boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int index = indices[i];

        LockToken<Pointer> lt = null;
        try {
            lt = NoisePatternCache.lockNoisePatternFreq(deviceNo, index, flipped);

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
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, flipped);
                    done[i] = true;
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, flipped);
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
    private void retrieveFrequencyDomainBoth(Constellation cons, int i, int[] indices, File[] files, boolean[] done)
            throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int index = indices[i];

        LockToken<Pointer> ltRegular = null;
        LockToken<Pointer> ltFlipped = null;
        try {
            ltRegular = NoisePatternCache.lockNoisePatternFreq(deviceNo, index, REGULAR);
            ltFlipped = NoisePatternCache.lockNoisePatternFreq(deviceNo, index, FLIPPED);
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
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, FLIPPED);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(deviceNo, index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, FLIPPED);
                    done[i] = false;
                }

            } else if (ltRegular.writeLock() && ltFlipped.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} flipped on the device", i, index);
                }

                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, REGULAR, device, ltRegular.availableElement);
                    setNoisePatternFreq(i, FLIPPED, ltFlipped.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreq(deviceNo, index, FLIPPED);
                    done[i] = false;
                }

            } else if (ltRegular.writeLock() && ltFlipped.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("none of noise patterns freq {} {} on the device", i, index);
                }

                if (noisePatternOnDevice(cons, index, files[i], device)) {
                    produceNoisePatternFreq(cons, i, index, FLIPPED, device, ltFlipped.availableElement);
                    produceNoisePatternFreq(cons, i, index, REGULAR, device, ltRegular.availableElement);
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, FLIPPED);
                    NoisePatternCache.toReadLockNoisePatternFreq(deviceNo, index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, FLIPPED);
                    done[i] = false;
                }
            } else {
                throw new Error("should not happen");
            }
        } catch (LockException e) {
            if (ltRegular != null) {
                if (ltRegular.writeLock()) {
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, REGULAR);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(deviceNo, index, REGULAR);
                }
            }
            if (ltFlipped != null) {
                if (ltFlipped.writeLock()) {
                    NoisePatternCache.unlockNoisePatternFreqRemove(deviceNo, index, FLIPPED);
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(deviceNo, index, FLIPPED);
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
                NoisePatternCache.unlockNoisePatternFreq(deviceNo, indicesI[i], true);
                for (int j = i + 1; j < indicesJ.length; j++) {
                    computeCorrelation(cons, i, j);
                }
                NoisePatternCache.unlockNoisePatternFreq(deviceNo, indicesI[i], false);
            }
        } else {
            for (int i = 0; i < indicesI.length; i++) {
                for (int j = 0; j < indicesJ.length; j++) {
                    computeCorrelation(cons, i, j);
                }
                NoisePatternCache.unlockNoisePatternFreq(deviceNo, indicesI[i], false);
            }
            for (int j = 0; j < indicesJ.length; j++) {
                NoisePatternCache.unlockNoisePatternFreq(deviceNo, indicesJ[j], true);
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

    public synchronized void pushMessage(Event event2, FetchPatternActivity a) {
        if (a == fI) {
            messageI = event2;
        } else {
            messageJ = event2;
        }
        notifyAll();
    }

}
