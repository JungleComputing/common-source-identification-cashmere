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

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.jocl.CL;
import org.jocl.CLException;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationCreationException;
import ibis.constellation.Context;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.OrContext;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.Timer;
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
import nl.junglecomputing.common_source_identification.mc.ExecutorData;
import nl.junglecomputing.common_source_identification.mc.FFT;

public class CommonSourceIdentification {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification");

    // The threshold for this node for device subdivision. This will depend on
    // the number of executes, the amount of memory on
    // the many-core device. Initially, we set it to a very conservative value.
    static int thresholdDC = 2;

    public static ActivityIdentifier progressActivity(SingleEventCollector sec, int nrImages) throws NoSuitableExecutorException {

        int nrCorrelations = ((nrImages - 1) * nrImages) / 2;

        ActivityIdentifier aid = Cashmere.submit(sec);

        ProgressActivity progressActivity = new ProgressActivity(aid, nrCorrelations);

        ActivityIdentifier progressActivityID = Cashmere.submit(progressActivity);

        return progressActivityID;
    }

    static ConstellationConfiguration[] getConfigurations() {
        int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);

        StealPool stealPool = new StealPool(NodeInformation.STEALPOOL);

        ConfigurationFactory configurationFactory = new ConfigurationFactory();
        // We create one executor for the main activities,
        // correlationMatrixActivity, single/multiple event collectors. Its pool
        // is stealPool and it does not steal from any steal pool.
        configurationFactory.createConfigurations(1, stealPool, StealPool.NONE, NodeInformation.LABEL);

        // We create one executor for the node activities with steal pool
        // stealPool from which it also steals. Note that the
        // label contains the hostname of this node, which means that this
        // executor will only steal activities with the label
        // that matches the hostname.
        configurationFactory.createConfigurations(1, stealPool, stealPool, NodeInformation.HOSTNAME + NodeActivity.LABEL);

        // We create nrLocalExecutors executors for CorrelationsActivities with
        // steal pool stealPool from which it also steals.
        // These executors either steal CorrelationsActivity for all nodes, or
        // specific to this node. The strategy is to steal
        // the biggest jobs locally and smallest jobs globally.
        configurationFactory.createConfigurations(nrLocalExecutors, stealPool, stealPool,
                new OrContext(new Context(NodeInformation.HOSTNAME + CorrelationsActivity.LABEL),
                        new Context(CorrelationsActivity.LABEL)),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        // executors for other small tasks that do not really divide or perform
        // work
        configurationFactory.createConfigurations(1, stealPool, stealPool, NodeInformation.HOSTNAME + NotifyParentActivity.LABEL);
        configurationFactory.createConfigurations(1, stealPool, StealPool.NONE, ProgressActivity.LABEL);
        logger.debug("creating configuration for: {}", NodeInformation.HOSTNAME + BarrierActivity.LABEL);
        configurationFactory.createConfigurations(1, stealPool, stealPool, NodeInformation.HOSTNAME + BarrierActivity.LABEL);

        return configurationFactory.getConfigurations();
    }

    static CorrelationMatrix submitCorrelation(SingleEventCollector sec, int height, int width, File[] imageFiles,
            List<String> nodes) throws NoSuitableExecutorException {
        CorrelationMatrixActivity correlationMatrixActivity = new CorrelationMatrixActivity(height, width, imageFiles, nodes);

        ActivityIdentifier aid = Cashmere.submit(sec);
        logger.debug("Submitting correlationActivity");
        Cashmere.submit(correlationMatrixActivity.setParent(aid));
        return (CorrelationMatrix) sec.waitForEvent().getData();
    }

    /*
     * Various Constellation activities
     */

    static void barrierNode() throws NoSuitableExecutorException {
        logger.debug("hostname: {}", NodeInformation.HOSTNAME);
        Cashmere.submit(new BarrierActivity(NodeInformation.HOSTNAME));
    }

    static void notifyNodesOfMaster(List<String> nodes, ActivityIdentifier progressActivityID)
            throws NoSuitableExecutorException {

        MultiEventCollector mec = new MultiEventCollector(new Context(NodeInformation.LABEL), nodes.size());

        ActivityIdentifier aid = Cashmere.submit(mec);

        for (String node : nodes) {
            Cashmere.submit(new NotifyParentActivity(node, aid, progressActivityID));
        }

        logger.debug("hiero");

        mec.waitForEvents();

        logger.debug("na wachten");

    }

    static void printImageToNodeMapping(File[] imageFiles, List<String> nodes) {
        if (logger.isDebugEnabled()) {
            int nrImagesPerNode = (int) Math.ceil((double) imageFiles.length / nodes.size());
            for (int i = 0; i < nodes.size(); i++) {
                int startIndex = i * nrImagesPerNode;
                int endIndex = Math.min(startIndex + nrImagesPerNode, imageFiles.length);

                for (int j = startIndex; j < endIndex; j++) {
                    logger.debug("Image {} is associated with {}", j, nodes.get(i));
                }
            }
        }
    }

    public static void main(String[] args) throws NoSuitableExecutorException {
        // every node in the cluster does the following:
        NodeInformation.setHostName();
        int nrNodes = 1;
        Version version = Version.DEVICE_MEM_CACHE;

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
            } else if (args[i].equals("-deviceMemCache")) {
            } else {
                throw new Error(nl.junglecomputing.common_source_identification.CommonSourceIdentification.USAGE);
            }
        }

        try {
            File[] imageFiles = IO.getImageFiles(nameImageDir);
            ImageDims imageDims = new ImageDims(imageFiles[0]);
            int height = imageDims.height;
            int width = imageDims.width;

            Cashmere.initialize(getConfigurations());
            Constellation constellation = Cashmere.getConstellation();

            int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);

            Device device = Cashmere.getDevice("grayscaleKernel");
            // we set the number of blocks for reduction operations to the
            // following value
            int nrBlocksForReduce = 1024;
            long memoryToBeReservedPerThread = ExecutorData.memoryForKernelExecutionThread(height, width, nrBlocksForReduce,
                    false);

            // we initialize for all nrLocalExecutors executors private data
            ExecutorData.initialize(nrLocalExecutors, device, height, width, nrBlocksForReduce, false);
            int nrNoisePatternsFreqDevice = CacheConfig.initializeCache(device, height, width,
                    memoryToBeReservedPerThread * nrLocalExecutors, nrLocalExecutors);

            // we compute a value for the threshold for subdivision of work
            // based on the number of noise patterns on the
            // device and the number of executors that have to share the
            // device.
            thresholdDC = nrNoisePatternsFreqDevice / 2 / nrLocalExecutors;
            logger.debug("{} parallel activities", nrLocalExecutors);
            logger.debug(String.format("Setting the d&c threshold to: %d/2/%d = %d", nrNoisePatternsFreqDevice, nrLocalExecutors,
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

            // every node starts a barrier activity
            barrierNode();

            if (constellation.isMaster()) {
                logger.info("CommonSourceIdentification, running with number of nodes: " + nrNodes);
                logger.info("image-dir: " + nameImageDir);

                logger.info("I am the master, my hostname is: {}, pid: {}", NodeInformation.HOSTNAME,
                        NodeInformation.getProcessId("<PID>"));

                printImageToNodeMapping(imageFiles, nodes);

                Timer timer = Cashmere.getOverallTimer();

                // we start a progress activity that notifies once in a while
                // how many of the total correlations have been done
                SingleEventCollector sec2 = new SingleEventCollector(new Context(NodeInformation.LABEL));
                ActivityIdentifier progressActivityID = progressActivity(sec2, imageFiles.length);

                /*
                 * The master sends out notifications of who the master is, the
                 * barrier will follow through. This means that we can start the
                 * timer and start computing.
                 */
                notifyNodesOfMaster(nodes, progressActivityID);

                // we will now start computing
                logger.info("Starting common source identification");
                int eventNo = timer.start();

                SingleEventCollector sec = new SingleEventCollector(new Context(NodeInformation.LABEL));
                CorrelationMatrix result = submitCorrelation(sec, height, width, imageFiles, nodes);
                ArrayList<Link> linkage = Linkage.hierarchical_clustering(result.coefficients);

                timer.stop(eventNo);
                // and we are done

                long timeNanos = (long) (timer.totalTimeVal() * 1000);
                System.out.println("Common source identification time: " + ProgressActivity.format(Duration.ofNanos(timeNanos)));

                // we wait for the progress activit to stop
                sec2.waitForEvent();

                // printTimings(nodes, timer.totalTimeVal());

                int n = imageFiles.length;
                // TODO: this has to be counted with a statistics
                int nrNoisePatternsComputed = (n * (n - 1)) / 2;
                int nrNoisePatternsTransformed = (n * (n - 1)) / 2;

                CountFLOPS.printGFLOPS(height, width, imageFiles.length, nrNoisePatternsComputed, nrNoisePatternsTransformed,
                        timeNanos);

                Timer writeFilesTimer = Cashmere.getTimer("java", "master", "Write files");
                int writeEvent = writeFilesTimer.start();
                IO.writeFiles(result, imageFiles, version);
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
            Cashmere.deinitializeLibraries();
            CacheConfig.clearCaches();
        } catch (IOException | ConstellationCreationException | CashmereNotAvailable e) {
            throw new Error(e);
        }
    }
}
