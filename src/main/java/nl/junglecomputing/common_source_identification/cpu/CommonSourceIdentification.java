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

package nl.junglecomputing.common_source_identification.cpu;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.ConstellationCreationException;
import ibis.constellation.Context;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.Timer;
import ibis.constellation.util.SingleEventCollector;
import nl.junglecomputing.common_source_identification.Version;

public class CommonSourceIdentification {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification");

    static ConstellationConfiguration[] getConfigurations() {
        int nrLocalExecutors = NodeInformation.getNrExecutors("cashmere.nLocalExecutors", 2);

        StealPool stealPool = new StealPool(NodeInformation.STEALPOOL);

        ConfigurationFactory configurationFactory = new ConfigurationFactory();

        configurationFactory.createConfigurations(nrLocalExecutors, stealPool, stealPool, new Context(CorrelationsActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

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
        Version version = Version.CPU;

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
            } else if (args[i].equals("-cpu")) {
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

                int n = imageFiles.length;
                int nrNoisePatternsComputed = (n * (n - 1)) / 2;
                int nrNoisePatternsTransformed = (n * (n - 1)) / 2;

                CountFLOPS.printGFLOPS(height, width, imageFiles.length, nrNoisePatternsComputed, nrNoisePatternsTransformed,
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

            System.exit(0);
        } catch (IOException | ConstellationCreationException e) {
            throw new Error(e);
        }
    }
}
