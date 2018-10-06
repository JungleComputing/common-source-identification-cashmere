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

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;

/*
 * The main activity that will produce the CorrelationMatrix
 */
class CorrelationMatrixActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationMatrixActivity");

    private ActivityIdentifier parent;

    private ArrayList<CorrelationsNodeActivity> correlationsNodeActivities;
    private CorrelationMatrix correlationMatrix;

    private List<String> nodes;

    private int nrReceivedCorrelations;

    CorrelationMatrixActivity(int h, int w, File[] imageFiles, List<String> nodes, boolean mc) {
        super(new Context(CommonSourceIdentification.LABEL), false, true);

        this.parent = null;
        this.correlationsNodeActivities = new ArrayList<CorrelationsNodeActivity>();

        this.correlationMatrix = new CorrelationMatrix(imageFiles.length);

        for (int i = 0; i < nodes.size(); i++) {
            correlationsNodeActivities.add(new CorrelationsNodeActivity(h, w, nodes, i, imageFiles, mc));
        }

        this.nodes = nodes;
        this.nrReceivedCorrelations = 0;
    }

    CorrelationMatrixActivity setParent(ActivityIdentifier aid) {
        this.parent = aid;
        return this;
    }

    @Override
    public void cleanup(Constellation constellation) {
        constellation.send(new Event(identifier(), parent, correlationMatrix));
    }

    @Override
    public int initialize(Constellation constellation) {
        try {
            submitCorrelationsNodeActivities(constellation);
        } catch (NoSuitableExecutorException e) {
            logger.error("Could not submit activity", e);
            return FINISH;
        }

        return SUSPEND;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        Object data = event.getData();

        if (data instanceof Correlations) {
            Correlations correlations = (Correlations) event.getData();
            return processCorrelations(correlations);
        } else {
            throw new Error("Unknown type of data");
        }

    }

    void submitCorrelationsNodeActivities(Constellation constellation) throws NoSuitableExecutorException {
        for (int i = 0; i < correlationsNodeActivities.size(); i++) {
            if (logger.isDebugEnabled()) {
                logger.debug("Submitting correlationsNodeActivity for {}", nodes.get(i));
            }
            constellation.submit(correlationsNodeActivities.get(i).setParent(identifier()));
        }
    }

    int processCorrelations(Correlations correlations) {
        logger.debug("Receiving correlations");

        for (Correlation correlation : correlations) {
            if (logger.isDebugEnabled()) {
                logger.debug("Adding correlation {}, {}", correlation.i, correlation.j);
            }

            correlationMatrix.add(correlation);
        }

        nrReceivedCorrelations++;

        if (nrReceivedCorrelations == nodes.size()) {
            return FINISH;
        } else {
            return SUSPEND;
        }
    }
}
