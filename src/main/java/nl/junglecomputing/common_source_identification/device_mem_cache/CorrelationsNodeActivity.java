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
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Constellation;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;

/*
 * Instances of this class are responsible for creating CorrelationsActivities.  As input they get their nodeIndex and compute
 * based on that which range of the images they are responsible for.  For example, if there are 100 images to be correlated, and
 * there are 4 nodes, and an instance has nodeIndex 2, then its range will be 50-75.  It will submit jobs for itself, in the
 * range (50-75, 50-75) that noone can steal, and it will create jobs that combine with ranges from other nodes, for example the
 * node with range 75-100.  We do this by computing the number of tiles of correlations that have to be computed, in this case
 * (100*99)/2, dividing it by the number of nodes, 4 in this case and make this node with index 2 responsible for the third part
 * (2+1).
 */
class CorrelationsNodeActivity extends NodeActivity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationsNodeActivity");

    // keeping track of the activities that we create/submit and the correlations that we receive
    private ArrayList<CorrelationsActivity> correlationsActivities;
    private int nrReceivedCorrelations;
    private Correlations correlations;

    CorrelationsNodeActivity(int h, int w, List<String> nodes, int nodeIndex, File[] imageFiles) {
        super(h, w, nodes, nodeIndex, imageFiles);

        correlationsActivities = new ArrayList<CorrelationsActivity>();
        this.correlations = new Correlations();
    }

    @Override
    public int initialize(Constellation cons) {
        createCorrelationsActivities();
        try {
            submitCorrelationsActivities(cons);
        } catch (NoSuitableExecutorException e) {
            logger.error("Could not submit activities", e);
            return FINISH;
        }
        return SUSPEND;
    }

    @Override
    public void cleanup(Constellation cons) {
        cons.send(new Event(identifier(), parent, correlations));
    }

    @Override
    public int process(Constellation cons, Event event) {
        Object data = event.getData();

        if (data instanceof Correlations) {
            return processCorrelations((Correlations) data);
        } else {
            throw new Error("Unknown type of data");
        }
    }

    // private methods

    private void createCorrelationsActivities() {
        logger.debug("Combining images of nodes");

        createCorrelationsActivitySelf();

        // determing the range of tiles this instance is responsible for and submitting activities for those tiles.
        int nrNodes = nodes.size();
        int nrTiles = nrNodes * (nrNodes - 1) / 2;
        int nrTilesPerNode = getNrTilesPerNode(nodeIndex, nrTiles, nrNodes);

        for (int i = 0; i < nrTilesPerNode; i++) {
            int tile = (nodeIndex + i + 1) % nrNodes;
            if (logger.isDebugEnabled()) {
                logger.debug("combine noise patterns from node {} with those from {}", nodeIndex, tile);
            }

            createCorrelationsActivity(tile, nrNodes);
        }
    }

    // the correlations (startIndex-endIndex, startIndex-endIndex) will be assigned to this node
    private void createCorrelationsActivitySelf() {
        correlationsActivities.add(new TreeCorrelationsActivity(startIndex, endIndex, startIndex, endIndex, nodeIndex, nodeIndex,
                h, w, nodes, imageFiles, 100));
    }

    private void createCorrelationsActivity(int tile, int nrNodes) {
        int node1 = tile < nodeIndex ? tile : nodeIndex;
        int node2 = tile < nodeIndex ? nodeIndex : tile;

        int startIndexI = getStartIndex(node1, nrImages, nrNodes);
        int endIndexI = getEndIndex(startIndexI, getNrImagesPerNode(nrImages, nrNodes), nrImages);
        int startIndexJ = getStartIndex(node2, nrImages, nrNodes);
        int endIndexJ = getEndIndex(startIndexJ, getNrImagesPerNode(nrImages, nrNodes), nrImages);

        int nrI = endIndexI - startIndexI;
        int nrJ = endIndexJ - startIndexJ;
        if (logger.isDebugEnabled()) {
            logger.debug("nrI: {}, nrJ: {}", nrI, nrJ);
        }

        if (nrI + nrJ <= CommonSourceIdentification.thresholdDC) {
            correlationsActivities.add(
                    new LeafCorrelationsActivity(startIndexI, endIndexI, startIndexJ, endIndexJ, h, w, nodes, imageFiles, 0));
        } else {
            correlationsActivities.add(new TreeCorrelationsActivity(startIndexI, endIndexI, startIndexJ, endIndexJ, node1, node2,
                    h, w, nodes, imageFiles, 0));
        }
    }

    private void submitCorrelationsActivities(Constellation cons) throws NoSuitableExecutorException {
        for (CorrelationsActivity correlationsActivity : correlationsActivities) {
            if (logger.isDebugEnabled()) {
                logger.debug("Submitting correlationsActivity");
                synchronized (TreeCorrelationsActivity.class) {
                    TreeCorrelationsActivity.submitted++;
                }
            }
            cons.submit(correlationsActivity.setParent(identifier()));
        }
    }

    private int processCorrelations(Correlations correlations) {
        nrReceivedCorrelations++;
        if (logger.isDebugEnabled()) {
            logger.debug("Processing Correlations, received {}/{}", nrReceivedCorrelations, correlationsActivities.size());
        }
        this.correlations.addAll(correlations);

        if (nrReceivedCorrelations == correlationsActivities.size()) {
            return FINISH;
        } else {
            return SUSPEND;
        }
    }

    private int getNrTilesPerNode(int nodeIndex, int nrTiles, int nrNodes) {
        if (nrTiles % nrNodes == 0) {
            return nrTiles / nrNodes;
        } else {
            if (nodeIndex < nrNodes / 2) {
                return nrTiles / nrNodes + 1;
            } else {
                return nrTiles / nrNodes;
            }
        }
    }
}
