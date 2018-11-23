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
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import org.jocl.CL;
import org.jocl.CLException;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationCreationException;
import ibis.constellation.ConstellationProperties;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.Timer;
import ibis.constellation.util.ByteBufferCache;
import ibis.constellation.util.MemorySizes;
import ibis.constellation.util.MultiEventCollector;
import ibis.constellation.util.SingleEventCollector;
import nl.junglecomputing.common_source_identification.Version;
import nl.junglecomputing.common_source_identification.cpu.ConfigurationFactory;
import nl.junglecomputing.common_source_identification.cpu.CorrelationMatrix;
import nl.junglecomputing.common_source_identification.cpu.CountFLOPS;
import nl.junglecomputing.common_source_identification.cpu.IO;
import nl.junglecomputing.common_source_identification.cpu.ImageDims;
import nl.junglecomputing.common_source_identification.cpu.JobSubmission;
import nl.junglecomputing.common_source_identification.cpu.Link;
import nl.junglecomputing.common_source_identification.cpu.Linkage;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;
import nl.junglecomputing.common_source_identification.device_mem_cache.CacheConfig;
import nl.junglecomputing.common_source_identification.device_mem_cache.NoisePatternCache;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;
import nl.junglecomputing.common_source_identification.mc.FFT;

public class CommonSourceIdentification {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification");

    // The threshold for this node for device subdivision. This will depend on
    // the number of executes, the amount of memory on
    // the many-core device. Initially, we set it to a very conservative value.
    static int thresholdDC = 2;

    private static int CHUNK_THRESHOLD = 48;

    /*
     * The following methods fill the nodes list with the hostnames of the nodes
     * in the cluster using the SGE job submission system.
     */

    static ConstellationConfiguration[] getConfigurations() {
        int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);
        int nrNoisePatternProviders = NodeInformation.getNrExecutors("np.providers", 2);

        StealPool stealPool = new StealPool(NodeInformation.STEALPOOL);
        StealPool localPool = new StealPool(NodeInformation.STEALPOOL + NodeInformation.ID);
        Context ctxt = new Context(CorrelationsActivity.LABEL + NodeInformation.ID);

        ConfigurationFactory configurationFactory = new ConfigurationFactory();

        configurationFactory.createConfigurations(nrLocalExecutors, StealPool.merge(stealPool, localPool), stealPool, ctxt,
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);
        configurationFactory.createConfigurations(1, stealPool, localPool, new Context(FetchPatternActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        if (NodeInformation.ID == 0) {
            // We only need the ones below on the master.
            // One thread for the progress activity
            configurationFactory.createConfigurations(1, stealPool, StealPool.NONE, new Context(ProgressActivity.LABEL),
                    StealStrategy.BIGGEST, StealStrategy.SMALLEST);
            // ... and one thread for the eventcollector(s)
            configurationFactory.createConfigurations(1, stealPool, StealPool.NONE, new Context(NodeInformation.LABEL),
                    StealStrategy.BIGGEST, StealStrategy.SMALLEST);
        }

        configurationFactory.createConfigurations(nrNoisePatternProviders, stealPool, stealPool,
                new Context(GetNoisePatternsActivity.LABEL + NodeInformation.ID), StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        return configurationFactory.getConfigurations();
    }

    public static int initializeCache(Device device, int height, int width, long toBeReserved, int nrThreads, int nImages) {
        int sizeNoisePattern = height * width * 4;
        int sizeNoisePatternFreq = sizeNoisePattern * 2;
        logger.info("Size of noise pattern: " + MemorySizes.toStringBytes(sizeNoisePattern));
        logger.info("Size of noise pattern freq: " + MemorySizes.toStringBytes(sizeNoisePatternFreq));

        int nrNoisePatternsFreqDevice = CacheConfig.getNrNoisePatternsDevice(sizeNoisePatternFreq, toBeReserved);
        long memReservedForGrayscale = height * width * 3 * nrThreads;

        int nByteBuffers = 3 * nrNoisePatternsFreqDevice / 4 + 1;
        logger.info("Reserving " + nByteBuffers + " bytebuffers for communication");
        ByteBufferCache.initializeByteBuffers(height * width * 4, nByteBuffers);
        // need memory for (de)serialization of byte buffers. We actually allocate a bit more than we will need,
        // to prevent the ByteBufferCache from allocating new buffers when a threshold is reached.
        memReservedForGrayscale += ((long) height) * width * 4 * nByteBuffers;

        int nrNoisePatternsMemory = Math.min(CacheConfig.getNrNoisePatternsMemory(sizeNoisePattern, memReservedForGrayscale),
                nImages);

        logger.info("memReservedForGrayscale = " + MemorySizes.toStringBytes(memReservedForGrayscale));
        logger.info("nrNoisePatternsMemory = " + nrNoisePatternsMemory);

        NoisePatternCache.initialize(device, height, width, nrNoisePatternsFreqDevice, nrNoisePatternsMemory);

        return nrNoisePatternsFreqDevice;
    }

    public static void clearCaches() {
        NoisePatternCache.clear();
    }

    static CorrelationMatrix submitCorrelations(SingleEventCollector sec, ActivityIdentifier id, ActivityIdentifier progress,
            int height, int width, List<String> nodes, int[][] lists, File[][] filesList, int nExecutors,
            ActivityIdentifier[][] providers) throws NoSuitableExecutorException {

        ArrayList<Activity> activities = new ArrayList<Activity>();

        int[][][] sublists = new int[lists.length][][];
        File[][][] subfilesLists = new File[lists.length][][];
        for (int i = 0; i < lists.length; i++) {
            int nchunksi = lists[i].length > CHUNK_THRESHOLD ? (lists[i].length + CHUNK_THRESHOLD - 1) / CHUNK_THRESHOLD : 1;
            int szi = nchunksi == 1 ? lists[i].length : CHUNK_THRESHOLD;
            sublists[i] = new int[nchunksi][];
            subfilesLists[i] = new File[nchunksi][];
            int offseti = 0;
            for (int k = 0; k < nchunksi; k++) {
                int limiti = Math.min(offseti + szi, lists[i].length);
                sublists[i][k] = Arrays.copyOfRange(lists[i], offseti, limiti);
                subfilesLists[i][k] = Arrays.copyOfRange(filesList[i], offseti, limiti);
                offseti = limiti;
            }
        }
        for (int i = 0; i < sublists.length; i++) {
            for (int j = i; j < sublists.length; j++) {
                for (int k = 0; k < sublists[i].length; k++) {
                    for (int l = 0; l < sublists[j].length; l++) {
                        if (i != j || l >= k) {
                            activities.add(new TreeCorrelationsActivity(id, progress, sublists[i][k], sublists[j][l], i, j,
                                    subfilesLists[i][k], subfilesLists[j][l], height, width, 1, providers));
                        }
                    }
                }
            }
        }
        Collections.shuffle(activities);
        for (Activity a : activities) {
            Cashmere.submit(a);
        }

        logger.debug("Submitted correlationActivities");
        return (CorrelationMatrix) sec.waitForEvent().getData();

    }

    public static void main(String[] args) throws NoSuitableExecutorException {
        // every node in the cluster does the following:
        NodeInformation.setHostName();
        int nrNodes = 1;

        String nt = System.getProperty("ibis.pool.size");
        if (nt != null) {
            nrNodes = Integer.parseInt(nt);
        }

        List<String> nodes = JobSubmission.getNodes();
        NodeInformation.setNodeID(nodes);
        String nameImageDir = "";

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-image-dir")) {
                i++;
                nameImageDir = args[i];
            } else if (args[i].equals("-dedicated-activities")) {
            } else {
                throw new Error(nl.junglecomputing.common_source_identification.CommonSourceIdentification.USAGE);
            }
        }

        try {
            File[] imageFiles = IO.getImageFiles(nameImageDir);
            ImageDims imageDims = new ImageDims(imageFiles[0]);
            int height = imageDims.height;
            int width = imageDims.width;

            // Make sure we know on which node the master resides.
            Properties p = System.getProperties();
            p.setProperty(ConstellationProperties.S_MASTER, NodeInformation.ID == 0 ? "true" : "false");

            Cashmere.initialize(getConfigurations(), p);
            Constellation constellation = Cashmere.getConstellation();

            Device device = Cashmere.getDevice("grayscaleKernel");
            int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);
            int nrNoisePatternProviders = NodeInformation.getNrExecutors("np.providers", 2);
            // we set the number of blocks for reduction operations to the
            // following value
            int nrBlocksForReduce = 1024;
            long memoryToBeReservedPerThread = ExecutorData.memoryForKernelExecutionThread(height, width, nrBlocksForReduce,
                    false);
            int nWorkers = nrLocalExecutors + nrNoisePatternProviders;

            // we initialize for all executors private data.
            // we need nrLocalExecutors+nrNoisePatternProviders of them,
            ExecutorData.initialize(nWorkers, device, height, width, nrBlocksForReduce, false);
            int nrNoisePatternsFreqDevice = initializeCache(device, height, width, memoryToBeReservedPerThread * nWorkers,
                    nWorkers, imageFiles.length);

            // we compute a value for the threshold for subdivision of work
            // based on the number of noise patterns on the
            // device and the number of executors that have to share the
            // device.
            thresholdDC = Math.max(nrNoisePatternsFreqDevice / nrLocalExecutors, 1);
            if (thresholdDC > 8) {
                CHUNK_THRESHOLD = thresholdDC;
            }
            logger.info("{} parallel activities", nrLocalExecutors);
            logger.info(String.format("Setting the d&c threshold to: %d/%d = %d", nrNoisePatternsFreqDevice, nrLocalExecutors,
                    thresholdDC));

            // we initialize the fft library for the many-core device
            Cashmere.setupLibrary("fft", (cl_context context, cl_command_queue queue) -> {
                int err = FFT.initializeFFT(context, queue, height, width);
                if (err != 0) {
                    throw new CLException(CL.stringFor_errorCode(err));
                }
            }, () -> {
                int err = FFT.deinitializeFFT();
                if (err != 0) {
                    throw new CLException(CL.stringFor_errorCode(err));
                }
            });
            Cashmere.initializeLibraries();
            constellation.activate();

            if (constellation.isMaster()) {
                // this is only executed by the master

                logger.info("CommonSourceIdentification, running with number of nodes: " + nrNodes
                        + ", nrNoisePatternProviders = " + nrNoisePatternProviders);
                logger.info("image-dir: " + nameImageDir);

                logger.info("I am the master, my hostname is: {}, pid: {}", NodeInformation.HOSTNAME,
                        NodeInformation.getProcessId("<PID>"));

                Timer timer = Cashmere.getOverallTimer();
                int eventNo = timer.start();
                logger.info("Starting common source identification");

                // Start activities for noise pattern providers.
                ActivityIdentifier[][] providers = new ActivityIdentifier[nrNodes][nrNoisePatternProviders];

                MultiEventCollector providerCollector = new MultiEventCollector(new Context(NodeInformation.LABEL),
                        nrNodes * nrNoisePatternProviders);
                ActivityIdentifier cid = Cashmere.submit(providerCollector);
                for (int i = 0; i < nrNodes; i++) {
                    for (int j = 0; j < nrNoisePatternProviders; j++) {
                        Activity provider = new GetNoisePatternsActivity(cid, height, width, i);
                        providers[i][j] = Cashmere.submit(provider);
                    }
                }
                providerCollector.waitForEvents();

                // Determine locations for time domain noise patterns.
                int[] locations;

                locations = new int[imageFiles.length];
                int sz = imageFiles.length / nrNodes;
                int remainder = imageFiles.length - sz * nrNodes;
                int currentNode = 0;
                int threshold = sz + (remainder > 0 ? 1 : 0);
                for (int i = 0; i < imageFiles.length; i++) {
                    if (i >= threshold) {
                        threshold += currentNode < remainder ? sz + 1 : sz;
                        currentNode++;
                    }
                    locations[i] = currentNode;
                }

                // Now we have locations for each noise pattern.
                // Create lists per node.
                int[][] lists = new int[nrNodes][locations.length];
                File[][] files = new File[nrNodes][locations.length];
                int[] sizes = new int[nrNodes];
                for (int i = 0; i < locations.length; i++) {
                    int node = locations[i];
                    files[node][sizes[node]] = imageFiles[i];
                    lists[node][sizes[node]++] = i;
                }
                for (int i = 0; i < nrNodes; i++) {
                    lists[i] = Arrays.copyOf(lists[i], sizes[i]);
                    files[i] = Arrays.copyOf(files[i], sizes[i]);
                }

                // Now submit correlation jobs for all pairs.

                // we start a progress activity that notifies once in a while
                // how many of the total correlations have been done
                SingleEventCollector sec = new SingleEventCollector(new Context(NodeInformation.LABEL));
                int nrImages = imageFiles.length;

                int nrCorrelations = ((nrImages - 1) * nrImages) / 2;

                ActivityIdentifier aid = Cashmere.submit(sec);
                CorrelationMatrixActivity cma = new CorrelationMatrixActivity(aid, nrImages);
                ActivityIdentifier collector = Cashmere.submit(cma);
                ProgressActivity progressActivity = new ProgressActivity(nrCorrelations);
                ActivityIdentifier progressActivityID = Cashmere.submit(progressActivity);

                CorrelationMatrix result = submitCorrelations(sec, collector, progressActivityID, height, width, nodes, lists,
                        files, nrLocalExecutors, providers);
                ArrayList<Link> linkage = Linkage.hierarchical_clustering(result.coefficients);

                timer.stop(eventNo);
                // and we are done

                long timeNanos = (long) (timer.totalTimeVal() * 1000);
                System.out.println("Common source identification time: " + ProgressActivity.format(Duration.ofNanos(timeNanos)));

                MultiEventCollector statisticsCollector = new MultiEventCollector(new Context(NodeInformation.LABEL), nrNodes);
                ActivityIdentifier sid = Cashmere.submit(statisticsCollector);
                for (int i = 0; i < nrNodes; i++) {
                    Activity getStats = new GetStatsActivity(sid, i, providers[i]);
                    Cashmere.submit(getStats);
                }

                Event[] events = statisticsCollector.waitForEvents();

                int nrNoisePatternsComputed = 0;
                int nrNoisePatternsTransformed = 0;
                int nrNoisePatternsSent = 0;

                for (int i = 0; i < events.length; i++) {
                    int[] stats = (int[]) events[i].getData();
                    nrNoisePatternsComputed += stats[0];
                    nrNoisePatternsTransformed += stats[1];
                    nrNoisePatternsSent += stats[2];
                }

                logger.info("nrNoisePatternsComputed = {}, nrNoisePatternsTransformed = {}, nrNoisePatternsSent = {}",
                        nrNoisePatternsComputed, nrNoisePatternsTransformed / 2, nrNoisePatternsSent);

                CountFLOPS.printGFLOPS(height, width, imageFiles.length, nrNoisePatternsComputed, nrNoisePatternsTransformed / 2,
                        timeNanos);

                Timer writeFilesTimer = Cashmere.getTimer("java", "master", "Write files");
                int writeEvent = writeFilesTimer.start();
                IO.writeFiles(result, imageFiles, Version.DEDICATED_ACTIVITIES);
                Linkage.write_linkage(linkage);
                Linkage.write_flat_clustering(linkage, imageFiles.length);
                writeFilesTimer.stop(writeEvent);
            } else {
                // we are a worker
                if (logger.isDebugEnabled()) {
                    logger.debug("I am a worker, my hostname is: {}, my pid is: {}", NodeInformation.HOSTNAME,
                            NodeInformation.getProcessId("<PID>"));
                }
            }

            // we are done, workers do this immediately but keep on stealing,
            // the master does this only when all activities have
            // ended.
            Cashmere.done();

            // cleanup
            clearCaches();
            Cashmere.deinitializeLibraries();

            // explicit exit because the FFT library sometimes keeps threads
            // alive preventing us to exit.
            System.exit(0);
        } catch (IOException | ConstellationCreationException | CashmereNotAvailable e) {
            throw new Error(e);
        }
    }
}
