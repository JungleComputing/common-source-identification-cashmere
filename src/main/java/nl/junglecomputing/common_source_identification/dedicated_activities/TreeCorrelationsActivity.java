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
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;
import nl.junglecomputing.common_source_identification.remote_activities.Correlations;

/*
 * TreeCorrelationsActivity instances will be subdivided into smaller instances depending on the threshold.  The threshold is
 * determined on initialization and depends on the number of executors and the memory that is available on the many-core device.
 * TreeCorrelationsActivities can be stolen by other nodes, which means that then the threshold may be different.  In the end,
 * they will split up into LeafCorrelationsActivities that cannot be stolen any longer and have to be executed by the node on
 * which they are created.
 */
public class TreeCorrelationsActivity extends CorrelationsActivity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.TreeCorrelationsActivity");
    static Logger memlogger = LoggerFactory.getLogger("CommonSourceIdentification.Cache");

    private String nodeCreatorActivity;

    // debugging, the number of TreeCorrelationsActivities in flight
    static int inFlight = 0;

    public TreeCorrelationsActivity(ActivityIdentifier parent, ActivityIdentifier progressActivityID, int[] indicesI,
            int[] indicesJ, int node1, int node2, File[] filesI, File[] filesJ, int h, int w, int level,
            ActivityIdentifier[][] providers) {
        super(parent, progressActivityID, indicesI, indicesJ, node1, node2, filesI, filesJ, h, w, level, providers);

        logger.debug("Creating TreeCorrelation, node1 = {}, node2 = {}, size1 = {}, size2 = {}, level = {}", node1, node2,
                indicesI.length, indicesJ.length, level);

        // the hostname of the node that created this
        this.nodeCreatorActivity = NodeInformation.HOSTNAME;
    }

    @Override
    public int initialize(Constellation cons) {

        if (logger.isDebugEnabled()) {
            logger.debug("Running TreeCorrelation, node1 = {}, node2 = {}, size1 = {}, size2 = {}", node1, node2, indicesI.length,
                    indicesJ.length);
            synchronized (TreeCorrelationsActivity.class) {
                inFlight++;
                logger.debug("{} in flight", inFlight);
            }
        }

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

        // thresholdDC contains the total number of images a tile may contain.
        int threshold = CommonSourceIdentification.thresholdDC / 2;

        if (nrActivitiesI / threshold > 2) {
            // The case where we can subdivide both ranges in two.  This will lead to a new TreeCorrelationsActivity.
            nrIterationsI = nrActivitiesI > threshold ? 2 : 1;
            nrIterationsJ = nrActivitiesJ > threshold ? 2 : 1;

            nrActivitiesPerIterI = getNrActivitiesPerIter(nrActivitiesI, nrIterationsI);
            nrActivitiesPerIterJ = getNrActivitiesPerIter(nrActivitiesJ, nrIterationsJ);
        } else {
            nrIterationsI = getNrIterations(nrActivitiesI, threshold);
            nrIterationsJ = getNrIterations(nrActivitiesJ, threshold);

            // Question now is, how large do we make the pieces? For instance, if the threshold is 15, but we enter with 16, do we make pieces of 15x15,1x1, 1x15, 15x1 or 4 times 8x8?

            // The case where we subdivide in pieces of size threshold. This is the first alternative.
            nrActivitiesPerIterI = threshold;
            nrActivitiesPerIterJ = threshold;

            // And this is the second alternative:
            // nrActivitiesPerIterI = getNrActivitiesPerIter(nrActivitiesI, nrIterationsI);
            // nrActivitiesPerIterJ = getNrActivitiesPerIter(nrActivitiesJ, nrIterationsJ);
        }

        // The following nested loop determines the start and end of the ranges and submits new Tree or
        // LeafCorrelationsActivities.
        for (int i = 0; i < nrIterationsI; i++) {
            int startIndexI = i * nrActivitiesPerIterI;
            int endIndexI = Math.min(startIndexI + nrActivitiesPerIterI, indicesI.length);

            for (int j = 0; j < nrIterationsJ; j++) {
                int startIndexJ = j * nrActivitiesPerIterJ;
                int endIndexJ = Math.min(startIndexJ + nrActivitiesPerIterJ, indicesJ.length);

                if (node1 != node2 || indicesI[0] != indicesJ[0]) {
                    // Ranges are certainly different
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
                } else if (startIndexI <= startIndexJ) {
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

    @Override
    public void cleanup(Constellation cons) {
        if (logger.isDebugEnabled()) {
            synchronized (TreeCorrelationsActivity.class) {
                inFlight--;
                logger.debug("{} in flight", inFlight);
            }
        }
        cons.send(new Event(identifier(), parent, correlations));
    }

    @Override
    public int process(Constellation cons, Event event) {
        Object data = event.getData();

        if (data instanceof Correlations) {
            Correlations correlations = (Correlations) data;
            return processCorrelations(correlations);
        } else {
            throw new Error("Unknown type of data");
        }
    }

    // private methods

    private boolean stolenActivity() {
        return !NodeInformation.HOSTNAME.equals(nodeCreatorActivity);
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

        int nrActivitiesI = indicesI.length;
        int nrActivitiesJ = indicesJ.length;

        CorrelationsActivity activity;
        if (nrActivitiesI + nrActivitiesJ <= CommonSourceIdentification.thresholdDC) {
            activity = new LeafCorrelationsActivity(identifier(), progressActivityID, indicesI, indicesJ, node1, node2, filesI,
                    filesJ, h, w, node1 == node2 ? 100 : level + 1, providers);
        } else {
            activity = new TreeCorrelationsActivity(identifier(), progressActivityID, indicesI, indicesJ, node1, node2, filesI,
                    filesJ, h, w, node1 == node2 ? 100 : level + 1, providers);
        }

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
}
