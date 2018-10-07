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

package nl.junglecomputing.common_source_identification.remote_activities;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import nl.junglecomputing.common_source_identification.cpu.Correlation;
import nl.junglecomputing.common_source_identification.cpu.CorrelationMatrix;

/*
 * The activity that will collect the CorrelationMatrix.
 */
public class CorrelationMatrixActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationMatrixActivity");

    private final ActivityIdentifier parent;
    private transient CorrelationMatrix correlationMatrix;

    private List<String> nodes;

    private transient int nrReceivedCorrelations;
    private transient int toReceive;
    private final int nImages;

    public CorrelationMatrixActivity(ActivityIdentifier parent, int nImages) {
        super(new Context(CommonSourceIdentification.LABEL), false, true);

        this.nImages = nImages;
        this.parent = parent;
    }

    @Override
    public void cleanup(Constellation constellation) {
        constellation.send(new Event(identifier(), parent, correlationMatrix));
    }

    @Override
    public int initialize(Constellation constellation) {
        nrReceivedCorrelations = 0;
        toReceive = nImages * (nImages - 1) / 2;
        correlationMatrix = new CorrelationMatrix(nImages);
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

    int processCorrelations(Correlations correlations) {
        logger.debug("Receiving correlations");

        for (Correlation correlation : correlations) {
            if (logger.isDebugEnabled()) {
                logger.debug("Adding correlation {}, {}", correlation.i, correlation.j);
            }

            correlationMatrix.add(correlation);
            nrReceivedCorrelations++;
        }

        if (nrReceivedCorrelations == toReceive) {
            return FINISH;
        } else {
            return SUSPEND;
        }
    }
}
