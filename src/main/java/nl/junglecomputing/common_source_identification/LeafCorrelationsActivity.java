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
import java.util.List;

import org.jocl.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.LibFuncNotAvailable;

import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Event;
import ibis.constellation.Timer;

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

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.LeafCorrelationsActivity");
    static Logger memlogger = LoggerFactory.getLogger("CommonSourceIdentification.Cache");

    static boolean FLIPPED = true;
    static boolean REGULAR = false;

    // identifies the activity that we notify the amount of correlations we have processed
    static ActivityIdentifier progressActivityID = null;

    // keeping track of the noise patterns
    private int nrNoisePatternsX;
    private Pointer[] noisePatternsXFreq;

    private int nrNoisePatternsY;
    private Pointer[] noisePatternsYFreq;

    /* 
     * The data that belongs to the executor that executes this activity.  Note that we cannot suspend this activity as it may be
     * picked up by a different executor.
     */
    private ExecutorData data;

    // keeping track of the time
    private int event;
    private Timer timer;

    // debugging, keeps track of the amount of the total LeafCorrelationsActivities that are in flight on the node.
    static int inFlight = 0;
    
    /*
     * Create a LeafCorrelationsActivity with the i and j range indicated by startI, endI, etc.
     */
    LeafCorrelationsActivity(int startI, int endI, int startJ, int endJ, int h, int w, List<String> nodes, File[] imageFiles, 
	    boolean mc, int level) {
	
        // made node2 node1 always equal, to trigger the fact that this should be local.
        super(startI, endI, startJ, endJ, CommonSourceIdentification.ID, CommonSourceIdentification.ID, h, w, nodes, imageFiles, mc, level);

        this.nrNoisePatternsX = endI - startI;
        this.noisePatternsXFreq = new Pointer[nrNoisePatternsX];

        this.nrNoisePatternsY = endJ - startJ;
        this.noisePatternsYFreq = new Pointer[nrNoisePatternsY];
    }


    @Override
    public int initialize(Constellation cons) {
        if (logger.isDebugEnabled()) {
            synchronized (LeafCorrelationsActivity.class) {
                inFlight++;
                logger.debug("{} in flight", inFlight);
            }
	    logger.debug("Executing LeafCorrelationsActivity for node {} by {}", nodeName1, CommonSourceIdentification.HOSTNAME);
            logger.debug(String.format("  (%d-%d),(%d-%d)", startI, endI, startJ, endJ));
        }

        String executor = cons.identifier().toString();
        this.timer = Cashmere.getTimer("java", executor, "leaf correlations");
        this.event = timer.start();

	if (mc) {
	    // we retrieve the data for this executor
	    data = ExecutorData.get(cons);
	    
	    try {
		// Since we want to produce many kernels on the same device, we pick the device based on some kernel
		Device device = Cashmere.getDevice("grayscaleKernel");

		boolean both = startI == startJ;

		if (both) {
		    // we need both the regular and flipped frequency domains as the i and j range overlap
		    retrieveFrequencyDomain(cons, startI, endI, false, both, device);
		} else {
		    retrieveFrequencyDomain(cons, startI, endI, REGULAR, both, device);
		    retrieveFrequencyDomain(cons, startJ, endJ, FLIPPED, both, device);
		}

		computeCorrelations(cons, startI, endI, startJ, endJ, device);

	    } catch (IOException | CashmereNotAvailable | LibFuncNotAvailable e) {
		throw new Error(e);
	    }
	} else {
	    try {
		ComputeCPU.computeCorrelations(correlations, h, w, imageFiles, startI, endI, startJ, endJ, executor);
	    } catch (IOException e) {
		throw new Error(e);
	    }
	}
	
        return FINISH;
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
                logger.debug("{} in flight", inFlight);
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
    private void retrieveFrequencyDomain(Constellation cons, int start, int end, boolean flipped, boolean both, Device device)
	throws CashmereNotAvailable, IOException, LibFuncNotAvailable {

	/* 
	 * We keep track of which in the range has already be done, since another thread may be processing it.  This means that 
	 * executors are cooperatively preparing the data.
	 */
        boolean[] done = new boolean[end - start];

        do {
            for (int i = start; i < end; i++) {
                retrieveFrequencyDomain(cons, i, start, flipped, both, device, done);
            }
        } while (!everythingDone(done));
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
    private void retrieveFrequencyDomain(Constellation cons, int index, int start, boolean flipped, boolean both, Device device,
            boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {
	
        int i = index - start;

        if (!done[i]) {
            if (!both) {
                retrieveFrequencyDomainSingle(cons, index, flipped, start, device, done);
            } else {
                retrieveFrequencyDomainBoth(cons, index, start, device, done);
            }
        }
    }

    /* 
     * Retrieves a frequency domain of a noise pattern indicated by index and flipped.  We try to lock the noise pattern and if
     * this succeeds then we may have read lock or write lock.  If we hold a read lock, then we are done, if we hold a write lock
     * we are supposed to produce the noise pattern.  Afterwards, we have to move the write lock to a read lock.
     *
     * pre:
     * - done[i] = false
     * - the current thread does not hold a lock for index
     * post:
     * - done[i] = true
     * - we hold a readlock of index flipped
     * - noisePatternsX/YFreq[i] is set
     */
    private void retrieveFrequencyDomainSingle(Constellation cons, int index, boolean flipped, int start, Device device, 
	    boolean[] done) throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int i = index - start;

        LockToken<Pointer> lt = null;
        try {
            lt = NoisePatternCache.lockNoisePatternFreq(index, flipped);

            if (lt.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} on the device", index, string(flipped));
                }
                setNoisePatternFreq(i, flipped, lt.availableElement);
                done[i] = true;
            } else if (lt.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} {} not on the device", index, string(flipped));
                }

		/*
		 * we are going to try to get the time domain noise pattern to the device.  If that succeeds, we produce the
		 * frequency domain noise pattern, otherwise, we did not succeed and we have to register that this noise pattern
		 * is not kept in the cache since it does not contain useful information.
		 */
                if (noisePatternOnDevice(cons, index, device)) {
                    produceNoisePatternFreq(cons, index, start, flipped, device, lt.availableElement);
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
            done[i] = false;
        }
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
    private boolean noisePatternOnDevice(Constellation cons, int index, Device device) throws CashmereNotAvailable, 
											      LibFuncNotAvailable, IOException {
        try {
            LockToken<Buffer> lt = NoisePatternCache.lockNoisePattern(index);
            if (lt.readLock()) {
                copyNoisePatternToDevice(index, device, lt.availableElement);
                NoisePatternCache.unlockNoisePattern(index);
                return true;
            } else if (lt.writeLock()) {
                produceNoisePattern(cons, index, device, lt.availableElement);
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
    private void copyNoisePatternToDevice(int index, Device device, Buffer noisePattern) {
        if (logger.isDebugEnabled()) {
            logger.debug("copying noise pattern {} to the device", index);
        }
	
        device.copy(noisePattern, data.noisePattern);
    }

    /*
     * pre:
     * - the current thread holds a write lock for noise pattern index
     * post:
     * - noise pattern index is on device
     * - noise pattern index is in the cache
    */
    private void produceNoisePattern(Constellation cons, int index, Device device, Buffer noisePattern)
	throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        if (logger.isDebugEnabled()) {
            logger.debug("computing noise pattern {}, putting it on the device", index);
        }

        ComputeNoisePattern.computePRNU_MC(index, imageFiles[index], h, w, cons.identifier().toString(), device, data);
	// get the data from the device, noisePattern points to memory in the cache
        device.get(noisePattern, data.noisePattern);
    }

    /*
     * pre:
     * - the current thread holds a write lock for noise pattern freq index
     * flipped
     * post:
     * - the device is associated with the index
     * - noisePatternsX/YFreq[i] is set
     */
    private void produceNoisePatternFreq(Constellation cons, int index, int start, boolean flipped, Device device, 
	    Pointer noisePatternFreq) throws CashmereNotAvailable, LibFuncNotAvailable {

        if (logger.isDebugEnabled()) {
            logger.debug("Computing frequency domain of {} {}, putting it on the device", index, string(flipped));
        }

        NoisePatternCache.setDevice(index, device);
        int i = index - start;
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
    private void retrieveFrequencyDomainBoth(Constellation cons, int index, int start, Device device, boolean[] done)
	throws CashmereNotAvailable, LibFuncNotAvailable, IOException {

        int i = index - start;

        LockToken<Pointer> ltRegular = null;
        LockToken<Pointer> ltFlipped = null;
        try {
            ltRegular = NoisePatternCache.lockNoisePatternFreq(index, REGULAR);
            ltFlipped = NoisePatternCache.lockNoisePatternFreq(index, FLIPPED);

            if (ltRegular.readLock() && ltFlipped.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("both noise patterns freq {} flipped and regular on the device", index);
                }
		
                setNoisePatternFreq(i, REGULAR, ltRegular.availableElement);
                setNoisePatternFreq(i, FLIPPED, ltFlipped.availableElement);
                done[i] = true;
		
            } else if (ltRegular.readLock() && ltFlipped.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} regular on the device", index);
                }
		
                if (noisePatternOnDevice(cons, index, device)) {
                    produceNoisePatternFreq(cons, index, start, FLIPPED, device, ltFlipped.availableElement);
                    setNoisePatternFreq(i, REGULAR, ltRegular.availableElement);
                    done[i] = true;
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreq(index, FLIPPED);
                    done[i] = false;
                }
		
            } else if (ltRegular.writeLock() && ltFlipped.readLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("noise pattern freq {} flipped on the device", index);
                }
		
                if (noisePatternOnDevice(cons, index, device)) {
                    produceNoisePatternFreq(cons, index, start, REGULAR, device, ltRegular.availableElement);
                    noisePatternsYFreq[i] = ltFlipped.availableElement;
                    setNoisePatternFreq(i, FLIPPED, ltFlipped.availableElement);
                    done[i] = true;
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreq(index, FLIPPED);
                    done[i] = false;
                }
		
            } else if (ltRegular.writeLock() && ltFlipped.writeLock()) {
                if (logger.isDebugEnabled()) {
                    logger.debug("none of noise patterns freq on the device");
                }
		
                if (noisePatternOnDevice(cons, index, device)) {
                    produceNoisePatternFreq(cons, index, start, FLIPPED, device, ltFlipped.availableElement);
                    produceNoisePatternFreq(cons, index, start, REGULAR, device, ltRegular.availableElement);
                    done[i] = true;
                } else {
                    NoisePatternCache.unlockNoisePatternFreq(index, REGULAR);
                    NoisePatternCache.unlockNoisePatternFreq(index, FLIPPED);
                    done[i] = false;
                }
            } else {
                throw new Error("should not happen");
            }
        } catch (LockException e) {
            if (ltRegular != null) {
                NoisePatternCache.unlockNoisePatternFreqRemove(index, REGULAR);
            }
            if (ltFlipped != null) {
                NoisePatternCache.unlockNoisePatternFreqRemove(index, FLIPPED);
            }
            done[i] = false;
        }
    }

    // correlations

    /*
     * Compute the correlations for the ranges.  We hold read locks for startI-endI regular and for startJ-endJ flipped.  We
     * unlock the read lock when done.
     */
    private void computeCorrelations(Constellation cons, int startI, int endI, int startJ, int endJ, Device device) 
	throws CashmereNotAvailable, LibFuncNotAvailable {


        if (startI == startJ /* && endI == endJ */) {
            for (int i = startI; i < endI; i++) {
                NoisePatternCache.unlockNoisePatternFreq(i, true);
                for (int j = i + 1; j < endJ; j++) {
                    computeCorrelation(cons, i, startI, j, startJ, device);
                }
                NoisePatternCache.unlockNoisePatternFreq(i, false);
            }
        } else {
            for (int i = startI; i < endI; i++) {
                for (int j = startJ; j < endJ; j++) {
                    computeCorrelation(cons, i, startI, j, startJ, device);
                }
                NoisePatternCache.unlockNoisePatternFreq(i, false);
            }
            for (int j = startJ; j < endJ; j++) {
                NoisePatternCache.unlockNoisePatternFreq(j, true);
            }
        }
    }

    private void computeCorrelation(Constellation cons, int i, int startI, int j, int startJ, Device device) 
	throws CashmereNotAvailable, LibFuncNotAvailable {

        if (logger.isDebugEnabled()) {
            logger.debug("Computing correlation of {},{}", i, j);
        }

        String executor = cons.identifier().toString();

        Correlation c = new Correlation(i, j);

        Pointer x = noisePatternsXFreq[i - startI];
        Pointer y = noisePatternsYFreq[j - startJ];

        c.coefficient = ComputeCorrelation.correlateMC(i, j, x, y, h, w, executor, device, data);
        correlations.add(c);
    }

    // common methods
    private boolean everythingDone(boolean[] done) {
        for (int i = 0; i < done.length; i++) {
            if (!done[i]) {
                return false;
            }
        }
        return true;
    }

    private String string(boolean flipped) {
        return flipped ? "flipped" : "regular";
    }
}
