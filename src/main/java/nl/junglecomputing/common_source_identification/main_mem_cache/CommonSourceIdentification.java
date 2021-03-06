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

package nl.junglecomputing.common_source_identification.main_mem_cache;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Device;
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationCreationException;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
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
import nl.junglecomputing.common_source_identification.cpu.ProgressActivity;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;
import nl.junglecomputing.common_source_identification.mc.FFT;

public class CommonSourceIdentification {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification");

    static ConstellationConfiguration[] getConfigurations() {
        int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);

        StealPool stealPool = new StealPool(NodeInformation.STEALPOOL);

        ConfigurationFactory configurationFactory = new ConfigurationFactory();

        configurationFactory.createConfigurations(nrLocalExecutors, stealPool, stealPool, new Context(CorrelationsActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        // We create one executor for the node activities with steal pool
        // stealPool from which it also steals. Note that the
        // label contains the hostname of this node, which means that this
        // executor will only steal activities with the label
        // that matches the hostname.
        configurationFactory.createConfigurations(1, stealPool, stealPool, NodeInformation.HOSTNAME + NodeInformation.LABEL);

        // One thread for the progress activity
        configurationFactory.createConfigurations(1, stealPool, stealPool, new Context(ProgressActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);
        // ... and one thread for the singleeventcollector.
        configurationFactory.createConfigurations(1, stealPool, stealPool, new Context(NodeInformation.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        return configurationFactory.getConfigurations();
    }

    static CorrelationMatrix submitCorrelations(SingleEventCollector sec, ActivityIdentifier id, int height, int width,
            File[] imageFiles, List<String> nodes) throws NoSuitableExecutorException {
        for (int i = 0; i < imageFiles.length; i++) {
            for (int j = i + 1; j < imageFiles.length; j++) {
                CorrelationsActivity ca = new CorrelationsActivity(id, height, width, i, j, imageFiles[i], imageFiles[j]);
                Cashmere.submit(ca);
            }
        }

        logger.debug("Submitted correlationActivities");
        return (CorrelationMatrix) sec.waitForEvent().getData();
    }

    public static void main(String[] args) throws NoSuitableExecutorException {
        // every node in the cluster does the following:
        NodeInformation.setHostName();
        int nrNodes = 1;
        Version version = Version.MAIN_MEM_CACHE;

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
            } else if (args[i].equals("-mainMemCache")) {
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
            CacheConfig.initializeCache(height, width, nrLocalExecutors);

            Device device = Cashmere.getDevice("grayscaleKernel");
            // we set the number of blocks for reduction operations to the
            // following value
            int nrBlocksForReduce = 1024;

            // we initialize for all nrLocalExecutors executors private data
            ExecutorData.initialize(nrLocalExecutors, device, height, width, nrBlocksForReduce, true);

            logger.debug("{} parallel activities", nrLocalExecutors);

            // we initialize the fft library for the many-core device
            Cashmere.setupLibrary("fft", (Device d, CommandStream queue) -> {
                FFT.initializeFFT(d, queue, height, width);
            }, () -> {
                FFT.deinitializeFFT();
            });
            Cashmere.initializeLibraries();
            constellation.activate();

            if (constellation.isMaster()) {
                logger.info("CommonSourceIdentification, running with number of nodes: " + nrNodes);
                logger.info("image-dir: " + nameImageDir);

                logger.info("I am the master, my hostname is: {}, pid: {}", NodeInformation.HOSTNAME,
                        NodeInformation.getProcessId("<PID>"));

                Timer timer = Cashmere.getOverallTimer();

                // we start a progress activity that notifies once in a while
                // how many of the total correlations have been done
                SingleEventCollector sec = new SingleEventCollector(new Context(NodeInformation.LABEL));
                ActivityIdentifier progressActivityID = NodeInformation.progressActivity(sec, imageFiles.length);

                // we will now start computing
                logger.info("Starting common source identification");
                int eventNo = timer.start();

                // start activities for all correlations.
                CorrelationMatrix result = submitCorrelations(sec, progressActivityID, height, width, imageFiles, nodes);
                ArrayList<Link> linkage = Linkage.hierarchical_clustering(result.coefficients);

                timer.stop(eventNo);
                // and we are done

                long timeNanos = (long) (timer.totalTimeVal() * 1000);
                System.out.println("Common source identification time: " + ProgressActivity.format(Duration.ofNanos(timeNanos)));

                MultiEventCollector statisticsCollector = new MultiEventCollector(new Context(NodeInformation.LABEL), nrNodes);
                ActivityIdentifier sid = Cashmere.submit(statisticsCollector);
                for (int i = 0; i < nrNodes; i++) {
                    Activity getStats = new GetStatsActivity(sid, nodes.get(i) + NodeInformation.LABEL);
                    Cashmere.submit(getStats);
                }

                Event[] events = statisticsCollector.waitForEvents();

                int nrNoisePatternsComputed = 0;
                int nrNoisePatternsTransformed = 0;

                for (int i = 0; i < events.length; i++) {
                    int[] stats = (int[]) events[i].getData();
                    nrNoisePatternsComputed += stats[0];
                    nrNoisePatternsTransformed += stats[1];
                }

                logger.info("nrNoisePatternsComputed = {}, nrNoisePatternsTransformed = {}", nrNoisePatternsComputed,
                        nrNoisePatternsTransformed / 2);

                CountFLOPS.printGFLOPS(height, width, imageFiles.length, nrNoisePatternsComputed, nrNoisePatternsTransformed / 2,
                        timeNanos);

                // we wait for the progress activity to stop
                sec.waitForEvent();

                // printTimings(nodes, timer.totalTimeVal());

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
