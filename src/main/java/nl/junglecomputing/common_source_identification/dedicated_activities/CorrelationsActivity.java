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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Context;
import ibis.constellation.OrContext;
import nl.junglecomputing.common_source_identification.cpu.NodeInformation;

// The base class for Leaf- and TreeCorrelationsActivity
abstract class CorrelationsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationsActivity");

    // Constellation logic
    static final String LABEL = "correlation";
    final protected ActivityIdentifier parent;

    // logic for subdivision of correlations
    protected int node1;
    protected int node2;

    protected int[] indicesI;
    protected int[] indicesJ;

    protected File[] filesI;
    protected File[] filesJ;

    protected int level;

    // Correlation logic
    protected int h;
    protected int w;

    protected Correlations correlations;
    protected int nrReceivedCorrelations;
    protected int nrCorrelationsToReceive;

    // identifies the activity that we notify the amount of correlations we have processed
    final protected ActivityIdentifier progressActivityID;

    final protected ActivityIdentifier[][] providers;

    CorrelationsActivity(boolean leaf, ActivityIdentifier parent, ActivityIdentifier progressActivityID, int[] indicesI,
            int[] indicesJ, int node1, int node2, File[] filesI, File[] filesJ, int h, int w, int level,
            ActivityIdentifier[][] providers) {
        /*
         * The images are subdivided over the nodes, for example, 0-25 to node A, 25-50 to B, 50-75 to C, and 75-100 to D.  The
         * correlations (0-25, 0-25) will be assigned to node A, (25-50, 25-50) to A or B, etc.  Then the correlations (0-25, 25-50)
         * could run efficiently on A and B since the images will likely to be there because of (0-25, 0-25) and (25-50, 25-50).
         */
        super(node1 == node2 ? new Context(LABEL + node1, level)
                : leaf ? new Context(LABEL + NodeInformation.ID, level)
                        : new OrContext(new Context(LABEL + node1, level), new Context(LABEL + node2, level)),
                true);

        logger.debug("Creating correlationsActivity with context " + getContext().toString() + ": " + indicesI.length + "x"
                + indicesJ.length);

        this.parent = parent;
        this.progressActivityID = progressActivityID;
        this.node1 = node1;
        this.node2 = node2;
        this.providers = providers;

        this.filesI = filesI;
        this.filesJ = filesJ;

        this.indicesI = indicesI;
        this.indicesJ = indicesJ;

        this.level = level;

        this.h = h;
        this.w = w;

        this.correlations = new Correlations();
        this.nrReceivedCorrelations = 0;
        this.nrCorrelationsToReceive = 0;
    }
}
