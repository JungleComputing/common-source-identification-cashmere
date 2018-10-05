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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
import ibis.constellation.ConstellationFactory;
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

import nl.junglecomputing.common_source_identification.Version;

@SuppressWarnings("restriction")
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
        final int height;
        final int width;

        ImageDims(File imageFile) throws IOException {
            BufferedImage image = FileToImageStage.readImage(imageFile);
            height = image.getHeight();
            width = image.getWidth();
        }

        ImageDims(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public boolean equals(Object other) {
            boolean result = false;
            if (other instanceof ImageDims) {
                ImageDims that = (ImageDims) other;
                result = that.canEqual(this) && that.height == this.height && that.width == this.width;
            }
            return result;
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(height, width);
        }

        public boolean canEqual(Object other) {
            return (other instanceof ImageDims);
        }
    }

    static final Map<ImageDims, Long> FFT_FLOPS_FORWARD = createFFTFlopsMap(true);
    static final Map<ImageDims, Long> FFT_FLOPS_BACKWARD = createFFTFlopsMap(false);


    /*
     * Functions for images and results
     */
    static Map<ImageDims, Long> createFFTFlopsMap(boolean forward) {
        HashMap<ImageDims, Long> map = new HashMap<ImageDims, Long>();
        if (forward) {
            map.put(new ImageDims(3000, 4000), 2351000000l);
        } else {
            map.put(new ImageDims(3000, 4000), 2352800000l);
        }
        return map;
    }

    static long nrFlopsGrayscale(int h, int w) {
        return 5 * h * w;
    }

    static long nrFlopsFastNoise(int h, int w) {
        return 2 * 3 * h * w;
    }

    static long nrFlopsZeroMean(int h, int w) {
        return 2 * (h * w * 2 + 2 * w);
    }

    static long nrFlopsWiener(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return flopsForwardFFT + h * w * 3 + h * w * 176 + (h * w * 3 + 2) + h * w * 4 + flopsBackwardFFT;
    }

    static long nrFlopsPRNU(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return nrFlopsGrayscale(h, w) + nrFlopsFastNoise(h, w) + nrFlopsZeroMean(h, w)
                + nrFlopsWiener(h, w, flopsForwardFFT, flopsBackwardFFT);
    }

    static long nrFlopsPCELinear(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return flopsForwardFFT * 2;
    }

    static long nrFlopsPCEQuadratic(int h, int w, long flopsForwardFFT, long flopsBackwardFFT) {
        return h * w * 6 + flopsBackwardFFT + h * w + (h - 11) * (h - 11) * 2 + 2;
    }

    static long nrFlops(int h, int w, int n, int nrNoisePatternsComputed, int nrNoisePatternFreqTransforms) {
        ImageDims imageDims = new ImageDims(h, w);
        long flopsForwardFFT = FFT_FLOPS_FORWARD.get(imageDims);
        long flopsBackwardFFT = FFT_FLOPS_BACKWARD.get(imageDims);

        long flopsPRNU = nrFlopsPRNU(h, w, flopsForwardFFT, flopsBackwardFFT);
        long flopsPCELinear = nrFlopsPCELinear(h, w, flopsForwardFFT, flopsBackwardFFT);
        long flopsPCEQuadratic = nrFlopsPCEQuadratic(h, w, flopsForwardFFT, flopsBackwardFFT);

        return nrNoisePatternsComputed * flopsPRNU + nrNoisePatternFreqTransforms * flopsPCELinear
                + ((n * (n - 1)) / 2) * flopsPCEQuadratic;
    }

    static void printGFLOPS(int h, int w, int n, int nrNoisePatternsComputed, int nrNoisePatternFreqTransforms, long timeNanos) {
        long nrFlopsAchieved = nrFlops(h, w, n, nrNoisePatternsComputed, nrNoisePatternFreqTransforms);
        long nrFlopsActual = nrFlops(h, w, n, n, n);
        double timeSeconds = timeNanos / 1e9;

        System.out.printf("achieved performance (counting everything computed): %.2f GFLOPS\n",
                nrFlopsAchieved / timeSeconds / 1e9);
        System.out.printf("actual performance (counting only what should be computed): %.2f GFLOPS\n",
                nrFlopsActual / timeSeconds / 1e9);
    }

    static ImageDims getImageDims(File imageFile) throws IOException {
        return new ImageDims(imageFile);
    }

    static void writeFiles(CorrelationMatrix correlationMatrix, File[] imageFiles, Version version) throws FileNotFoundException {
        PrintStream out = new PrintStream("prnu_" + version + ".out");

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
        Version version = Version.CPU;

        String nt = System.getProperty("ibis.pool.size");
        if (nt != null) {
            nrNodes = Integer.parseInt(nt);
        }

        List<String> nodes = getNodes();
        setNodeID(nodes);
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
            File[] imageFiles = getImageFiles(nameImageDir);
            ImageDims imageDims = getImageDims(imageFiles[0]);
            int height = imageDims.height;
            int width = imageDims.width;

            Cashmere.initialize(getConfigurations());
            Constellation constellation = Cashmere.getConstellation();

            constellation.activate();

            if (constellation.isMaster()) {
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
                CorrelationMatrix result = submitCorrelations(sec, progressActivityID, height, width, imageFiles, nodes);
                ArrayList<Link> linkage = Linkage.hierarchical_clustering(result.coefficients);

                timer.stop(eventNo);
                // and we are done

                long timeNanos = (long) (timer.totalTimeVal() * 1000);
                System.out.println("Common source identification time: " + ProgressActivity.format(Duration.ofNanos(timeNanos)));
                int n = imageFiles.length;
                int nrNoisePatternsComputed = (n * (n - 1)) / 2;
                int nrNoisePatternsTransformed = (n * (n - 1)) / 2;

                printGFLOPS(height, width, imageFiles.length, nrNoisePatternsComputed, nrNoisePatternsTransformed, timeNanos);

                // we wait for the progress activity to stop
                sec.waitForEvent();

                // printTimings(nodes, timer.totalTimeVal());

                Timer writeFilesTimer = Cashmere.getTimer("java", "master", "Write files");
                int writeEvent = writeFilesTimer.start();
                writeFiles(result, imageFiles, version);
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
	    constellation.done();

            System.exit(0);
        } catch (IOException | ConstellationCreationException e) {
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
}
