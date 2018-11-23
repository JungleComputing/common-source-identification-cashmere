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
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Constellation;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;

/*
 * TreeCorrelationsActivity instances will be subdivided into smaller instances depending on the threshold.  The threshold is
 * determined on initialization and depends on the number of executors and the memory that is available on the many-core device.
 * TreeCorrelationsActivities can be stolen by other nodes, which means that then the threshold may be different.  In the end,
 * they will split up into LeafCorrelationsActivities that cannot be stolen any longer and have to be executed by the node on
 * which they are created.
 */
class TreeCorrelationsActivity extends CorrelationsActivity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.TreeCorrelationsActivity");
    static Logger memlogger = LoggerFactory.getLogger("CommonSourceIdentification.Cache");

    private String nodeCreatorActivity;

    // debugging, the amount of TreeCorrelationsActivity in flight and submitted on this node
    static int inFlight = 0;
    static int submitted = 0;

    TreeCorrelationsActivity(int startI, int endI, int startJ, int endJ, int node1, int node2, int h, int w, List<String> nodes,
            File[] imageFiles, int level) {
        super(startI, endI, startJ, endJ, node1, node2, h, w, nodes, imageFiles, level);

        // the hostname of the node that created this
        this.nodeCreatorActivity = NodeInformation.HOSTNAME;
    }

    @Override
    public int initialize(Constellation cons) {

        if (logger.isDebugEnabled()) {
            synchronized (TreeCorrelationsActivity.class) {
                inFlight++;
                logger.debug("{} in flight of {} submitted", inFlight, submitted);
            }

            String node;
            if (node1 == node2) {
                node = nodeName1;
            } else {
                node = "everybody";
            }

            logger.debug(String.format("Executing for %s by %s:" + "  (%d-%d), (%d-%d)", node, NodeInformation.HOSTNAME, startI,
                    endI, startJ, endJ));
            if (stolenActivity()) {
                logger.debug("Executing a stolen activity");
            } else {
                logger.debug("Executing my activity");
            }
        }

        /* The following code splits up the range in the i direction and the range in the j direction based on the threshold.
         * There is a nested for loop that will submit the nested CorrelationsActiviities.
         */

        // determining the number of activities for i and j
        int nrActivitiesI = endI - startI;
        int nrActivitiesJ = endJ - startJ;

        // The following code determines the number of iterations of the nested for-loop and the number of activities each
        // subdivision will contain
        int nrIterationsI;
        int nrIterationsJ;

        int nrActivitiesPerIterI;
        int nrActivitiesPerIterJ;

        // thresholdDC contains the total number of images a tile may contain.
        int threshold = CommonSourceIdentification.thresholdDC / 2;

        if (nrActivitiesI / threshold > 2 || nrActivitiesJ / threshold > 2) {
            // The case where we can subdivide both ranges in two.  This will lead to a new TreeCorrelationsActivity.
            nrIterationsI = nrActivitiesI > threshold ? 2 : 1;
            nrIterationsJ = nrActivitiesJ > threshold ? 2 : 1;

            nrActivitiesPerIterI = getNrActivitiesPerIter(nrActivitiesI, nrIterationsI);
            nrActivitiesPerIterJ = getNrActivitiesPerIter(nrActivitiesJ, nrIterationsJ);
        } else {
            // The case where we subdivide in pieces of size threshold.
            nrActivitiesPerIterI = threshold;
            nrActivitiesPerIterJ = threshold;

            nrIterationsI = getNrIterations(nrActivitiesI, threshold);
            nrIterationsJ = getNrIterations(nrActivitiesJ, threshold);
        }

        // The following nested loop determines the start and end of the ranges and submits new Tree or
        // LeafCorrelationsActivities.
        for (int i = 0; i < nrIterationsI; i++) {
            int startIndexI = startI + i * nrActivitiesPerIterI;
            int endIndexI = Math.min(startIndexI + nrActivitiesPerIterI, endI);

            for (int j = 0; j < nrIterationsJ; j++) {
                int startIndexJ = startJ + j * nrActivitiesPerIterJ;
                int endIndexJ = Math.min(startIndexJ + nrActivitiesPerIterJ, endJ);

                if (startIndexI <= startIndexJ) {
                    try {
                        submitActivity(cons, startIndexI, endIndexI, startIndexJ, endIndexJ);
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
                logger.debug("{} in flight of {} submitted", inFlight, submitted);
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

    private void submitActivity(Constellation cons, int startIndexI, int endIndexI, int startIndexJ, int endIndexJ)
            throws NoSuitableExecutorException {

        int nrActivitiesI = endIndexI - startIndexI;
        int nrActivitiesJ = endIndexJ - startIndexJ;

        CorrelationsActivity activity;
        String kind;
        if (nrActivitiesI + nrActivitiesJ <= CommonSourceIdentification.thresholdDC) {
            // the LeafCorrelationsActivitiy will be assigned to this node
            activity = new LeafCorrelationsActivity(startIndexI, endIndexI, startIndexJ, endIndexJ, h, w, nodes, imageFiles,
                    level + 1);
            kind = "Leaf";
        } else {
            // we assign the LeafCorrelationsActivitiy to this node to either node1 or node2
            activity = new TreeCorrelationsActivity(startIndexI, endIndexI, startIndexJ, endIndexJ, node1, node2, h, w, nodes,
                    imageFiles, level + 1);
            kind = "Tree";
            synchronized (TreeCorrelationsActivity.class) {
                submitted++;
            }
        }
        activity.setParent(identifier());

        if (logger.isDebugEnabled()) {
            logger.debug(String.format("Submitting %s: (%d-%d), (%d-%d)", kind, startIndexI, endIndexI, startIndexJ, endIndexJ));
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
