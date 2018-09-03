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

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
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
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;
import ibis.constellation.Timer;
import ibis.constellation.util.MemorySizes;
import ibis.constellation.util.SingleEventCollector;
import sun.misc.VM;

public class CommonSourceIdentification {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification");

    // constants for setting up Constellation (some are package private)
    static String HOSTNAME = "localhost";
    static int ID = 0;
    private static String STEALPOOL = "stealpool";
    static String LABEL = "commonSourceIdentification";

    // simple class to return the height and width of the images that we are
    // correlating.
    static class ImageDims {
        int height;
        int width;

        ImageDims(File imageFile) throws IOException {
            BufferedImage image = FileToImageStage.readImage(imageFile);
            height = image.getHeight();
            width = image.getWidth();
        }
    }

    /*
     * Functions for images and results
     */
    static ImageDims getImageDims(File imageFile) throws IOException {
        return new ImageDims(imageFile);
    }

    static void writeFiles(CorrelationMatrix correlationMatrix, File[] imageFiles, boolean runOnMc) throws FileNotFoundException {
        String cpuOrMc = runOnMc ? "mc" : "cpu";

        PrintStream out = new PrintStream("prnu_" + cpuOrMc + ".out");

        double[][] coefficients = correlationMatrix.coefficients;

        for (int i = 0; i < coefficients.length; i++) {
            for (int j = 0; j < coefficients.length; j++) {
                out.printf("%.6f, ", coefficients[i][j]);
            }
            out.println();
        }
        out.close();
    }

    static File[] getImageFiles(String nameImageDir) throws IOException {
        File imageDir = new File(nameImageDir);
        if (!(imageDir.exists() && imageDir.isDirectory())) {
            throw new IOException(nameImageDir + " is not a valid directory");
        }
        File[] imageFiles = imageDir.listFiles();
        Arrays.sort(imageFiles, new Comparator<File>() {
            public int compare(File f1, File f2) {
                return f1.getName().compareTo(f2.getName());
            }
        });
        return imageFiles;
    }

    /*
     * The following methods that fill the nodes list with the hostnames of the
     * nodes in the cluster using the Slurm job submission system.
     */

    static void parseRangeSlurm(List<String> nodes, String prefix, String inputNodes) {
        String[] nodeNumbers = inputNodes.split("-");

        if (nodeNumbers.length == 1) {
            nodes.add(prefix + nodeNumbers[0]);
        } else {
            for (int i = 0; i < nodeNumbers.length; i += 2) {
                int start = Integer.parseInt(nodeNumbers[i]);
                int end = Integer.parseInt(nodeNumbers[i + 1]);

                for (int j = start; j <= end; j++) {
                    nodes.add(String.format("%s%03d", prefix, j));
                }
            }
        }
    }

    static void parseListSlurm(List<String> nodes, String prefix, String inputNodes) {
        String[] ranges = inputNodes.split(",");

        for (String range : ranges) {
            parseRangeSlurm(nodes, prefix, range);
        }
    }

    static void parseSuffixSlurm(List<String> nodes, String prefix, String inputNodes) {
        if (inputNodes.charAt(0) == '[') {
            parseListSlurm(nodes, prefix, inputNodes.substring(1, inputNodes.length() - 1));
        } else {
            nodes.add(prefix + inputNodes);
        }
    }

    static void parseNodesSlurm(List<String> nodes, String inputNodes) {
        if (inputNodes.startsWith("node")) {
            parseSuffixSlurm(nodes, "node", inputNodes.substring(4));
        }
    }

    /*
     * The following methods fill the nodes list with the hostnames of the nodes
     * in the cluster using the SGE job submission system.
     */

    static void parseNodesSGE(List<String> nodes) {
        String inputNodes = System.getenv("PRUN_HOSTNAMES");
        for (String s : inputNodes.split(" ")) {
            nodes.add(s.substring(0, 7));
        }
    }

    static List<String> getNodes() {
        List<String> nodes = new ArrayList<String>();

        String inputNodes = System.getenv("SLURM_JOB_NODELIST");
        if (inputNodes == null) {
            parseNodesSGE(nodes);
        } else {
            parseNodesSlurm(nodes, inputNodes);
        }
        if (nodes.isEmpty()) {
            nodes.add(HOSTNAME);
        }

        return nodes;
    }

    /*
     * All kinds of bookkeeping methods
     */

    static void setNodeID(List<String> nodes) {
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).equals(HOSTNAME)) {
                ID = i;
                return;
            }
        }
    }

    static int getNrExecutors(String property, int defaultValue) {
        String prop = System.getProperties().getProperty(property);
        int nrExecutors = defaultValue;
        if (prop != null) {
            nrExecutors = Integer.parseInt(prop);
        }

        return nrExecutors;
    }

    static void setHostName() {
        try {
            Runtime r = Runtime.getRuntime();
            Process p = r.exec("hostname");
            p.waitFor();
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            HOSTNAME = b.readLine();
            b.close();
        } catch (IOException | InterruptedException e) {
        }
    }

    static ConstellationConfiguration[] getConfigurations() {
        int nrLocalExecutors = getNrExecutors("cashmere.nLocalExecutors", 2);

        StealPool stealPool = new StealPool(STEALPOOL);

        ConfigurationFactory configurationFactory = new ConfigurationFactory();

        configurationFactory.createConfigurations(nrLocalExecutors, stealPool, stealPool, new Context(CorrelationsActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);

        // One thread for the progress activity
        configurationFactory.createConfigurations(1, stealPool, stealPool, new Context(ProgressActivity.LABEL),
                StealStrategy.BIGGEST, StealStrategy.SMALLEST);
        // ... and one thread for the singleeventcollector.
        configurationFactory.createConfigurations(1, stealPool, stealPool, new Context(LABEL), StealStrategy.BIGGEST,
                StealStrategy.SMALLEST);

        return configurationFactory.getConfigurations();
    }

    static CorrelationMatrix submitCorrelations(SingleEventCollector sec, ActivityIdentifier id, int height, int width,
            File[] imageFiles, List<String> nodes, boolean runOnMc, boolean useCache) throws NoSuitableExecutorException {
        for (int i = 0; i < imageFiles.length; i++) {
            for (int j = i + 1; j < imageFiles.length; j++) {
                CorrelationsActivity ca = new CorrelationsActivity(id, height, width, i, j, imageFiles[i], imageFiles[j], runOnMc,
                        useCache);
                Cashmere.submit(ca);
            }
        }

        logger.debug("Submitted correlationActivities");
        return (CorrelationMatrix) sec.waitForEvent().getData();
    }

    /*
     * Various Constellation activities
     */

    static ActivityIdentifier progressActivity(SingleEventCollector sec, int nrImages) throws NoSuitableExecutorException {

        ActivityIdentifier aid = Cashmere.submit(sec);

        ProgressActivity progressActivity = new ProgressActivity(aid, nrImages);

        ActivityIdentifier progressActivityID = Cashmere.submit(progressActivity);

        return progressActivityID;
    }

    static String getProcessId(final String fallback) {
        // Note: may fail in some JVM implementations
        // therefore fallback has to be provided

        // something like '<pid>@<hostname>', at least in SUN / Oracle JVMs
        final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
        final int index = jvmName.indexOf('@');

        if (index < 1) {
            // part before '@' empty (index = 0) / '@' not found (index = -1)
            return fallback;
        }

        try {
            return Long.toString(Long.parseLong(jvmName.substring(0, index)));
        } catch (NumberFormatException e) {
            // ignore
        }
        return fallback;
    }

    public static void main(String[] args) throws NoSuitableExecutorException {
        // every node in the cluster does the following:
        setHostName();
        int nrNodes = 1;
        boolean runOnMc = false;
        boolean useCache = false;

        String nt = System.getProperty("ibis.pool.size");
        if (nt != null) {
            nrNodes = Integer.parseInt(nt);
        }

        List<String> nodes = getNodes();
        setNodeID(nodes);
        String nameImageDir = "";

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-nrNodes")) {
                i++;
                nrNodes = Integer.parseInt(args[i]);
            } else if (args[i].equals("-image-dir")) {
                i++;
                nameImageDir = args[i];
            } else if (args[i].equals("-mc")) {
                runOnMc = true;
            } else if (args[i].equals("-useCache")) {
                useCache = true;
            } else if (args[i].equals("-cpu")) {
                runOnMc = false;
            } else {
                throw new Error("Usage: java CommonSourceIdentification -image-dir <image-dir> [ -cpu | -mc ]");
            }
        }

        try {
            File[] imageFiles = getImageFiles(nameImageDir);
            ImageDims imageDims = getImageDims(imageFiles[0]);
            int height = imageDims.height;
            int width = imageDims.width;

            Cashmere.initialize(getConfigurations());
            Constellation constellation = Cashmere.getConstellation();

            if (runOnMc) {
                // if we are running with many-cores enabled

                int nrLocalExecutors = getNrExecutors("cashmere.nLocalExecutors", 2);

                if (useCache) {
                    initializeCache(height, width, nrLocalExecutors);
                }

                Device device = Cashmere.getDevice("grayscaleKernel");
                // we set the number of blocks for reduction operations to the
                // following value
                int nrBlocksForReduce = 1024;

                // we initialize for all nrLocalExecutors executors private data
                ExecutorData.initialize(nrLocalExecutors, device, height, width, nrBlocksForReduce);

                logger.debug("{} parallel activities", nrLocalExecutors);

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
            }

            if (constellation.isMaster()) {
                // this is only executed by the master

                logger.info("CommonSourceIdentification, running with number of nodes: " + nrNodes);
                logger.info("image-dir: " + nameImageDir);

                logger.info("I am the master, my hostname is: {}, pid: {}", HOSTNAME, getProcessId("<PID>"));

                Timer timer = Cashmere.getOverallTimer();

                // we start a progress activity that notifies once in a while
                // how many of the total correlations have been done
                SingleEventCollector sec = new SingleEventCollector(new Context(LABEL));
                ActivityIdentifier progressActivityID = progressActivity(sec, imageFiles.length);

                // we will now start computing
                logger.info("Starting common source identification");
                int eventNo = timer.start();

                // start activities for all correlations.

                CorrelationMatrix result = submitCorrelations(sec, progressActivityID, height, width, imageFiles, nodes, runOnMc,
                        useCache);
                ArrayList<Link> linkage = Linkage.hierarchical_clustering(result.coefficients);

                timer.stop(eventNo);
                // and we are done

                long timeNanos = (long) (timer.totalTimeVal() * 1000);
                System.out.println("Common source identification time: " + ProgressActivity.format(Duration.ofNanos(timeNanos)));

                // we wait for the progress activit to stop
                sec.waitForEvent();

                // printTimings(nodes, timer.totalTimeVal());

                Timer writeFilesTimer = Cashmere.getTimer("java", "master", "Write files");
                int writeEvent = writeFilesTimer.start();
                writeFiles(result, imageFiles, runOnMc);
                Linkage.write_linkage(linkage);
                Linkage.write_flat_clustering(linkage, imageFiles.length);
                writeFilesTimer.stop(writeEvent);
            } else {
                // we are a worker
                if (logger.isDebugEnabled()) {
                    logger.debug("I am a worker, my hostname is: {}, my pid is: {}", HOSTNAME, getProcessId("<PID>"));
                }
            }

            // we are done, workers do this immediately but keep on stealing,
            // the master does this only when all activities have
            // ended.
            Cashmere.done();

            // cleanup
            if (runOnMc) {
                Cashmere.deinitializeLibraries();
                if (useCache) {
                    clearCaches();
                }
            }

            // explicit exit because the FFT library sometimes keeps threads
            // alive preventing us to exit.
            System.exit(0);
        } catch (IOException | ConstellationCreationException | CashmereNotAvailable e) {
            throw new Error(e);
        }
    }

    static int nrNoisePatternsForSpace(long space, long sizeNoisePattern) {
        return (int) Math.floor(space / (double) sizeNoisePattern);
    }

    static int getNrNoisePatternsMemory(int sizeNoisePattern, int spaceForGrayscale) {
        long spaceForNoisePatterns = VM.maxDirectMemory() - spaceForGrayscale;
        int nrNoisePatterns = nrNoisePatternsForSpace(spaceForNoisePatterns, sizeNoisePattern);

        if (logger.isDebugEnabled()) {
            logger.debug("space for noise patterns: " + MemorySizes.toStringBytes(spaceForNoisePatterns));
            logger.debug(String.format("The memory will hold a maximum of " + "%d noise patterns", nrNoisePatterns));
        }

        return nrNoisePatterns;
    }

    static void initializeCache(int height, int width, int nrThreads) {
        int sizeNoisePattern = height * width * 4;

        if (logger.isDebugEnabled()) {
            logger.debug("Size of noise pattern: " + MemorySizes.toStringBytes(sizeNoisePattern));
        }

        int memReservedForGrayscale = height * width * 3 * nrThreads;
        int nrNoisePatternsMemory = getNrNoisePatternsMemory(sizeNoisePattern, memReservedForGrayscale);

        NoisePatternCache.initialize(height, width, nrNoisePatternsMemory);

    }

    static void clearCaches() {
        NoisePatternCache.clear();
    }

}
