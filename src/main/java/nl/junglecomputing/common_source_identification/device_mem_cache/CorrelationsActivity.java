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

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Context;

// The base class for Leaf- and TreeCorrelationsActivity
abstract class CorrelationsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.CorrelationsActivity");

    // Constellation logic
    static final String LABEL = "correlation";
    protected ActivityIdentifier parent;

    // logic for subdivision of correlations
    protected List<String> nodes;
    protected int node1;
    protected int node2;
    protected String nodeName1;
    protected String nodeName2;
    protected File[] imageFiles;

    protected int startI;
    protected int endI;
    protected int startJ;
    protected int endJ;

    protected int level;

    // Correlation logic
    protected int h;
    protected int w;

    protected Correlations correlations;
    protected int nrReceivedCorrelations;
    protected int nrCorrelationsToReceive;

    CorrelationsActivity(int startI, int endI, int startJ, int endJ, int node1, int node2, int h, int w, List<String> nodes,
            File[] imageFiles, int level) {
        /*
         * The images are subdivided over the nodes, for example, 0-25 to node A, 25-50 to B, 50-75 to C, and 75-100 to D.  The
         * correlations (0-25, 0-25) will be assigned to node A, (25-50, 25-50) to B, etc.  Then the correlations (0-25, 25-50)
         * could run efficiently on A and B since the images will likely to be there because of (0-25, 0-25) and (25-50, 25-50).
         *
         * We could make sure that (0-25, 25-50) can only run on A and B, but for load-balancing purposes it is better to assign
         * them to any node.  However, we do make sure that CorrelationsActivities with the same range in the i and j direction,
         * so for example (0-25,0-25) gets assigned to a specific node.  We arrange that in the call to super() below:
         */
        super(node1 == node2 ? new Context(nodes.get(node1) + LABEL, level) : new Context(LABEL, level), true);

        this.nodes = nodes;
        this.node1 = node1;
        this.node2 = node2;
        this.nodeName1 = nodes.get(node1);
        this.nodeName2 = nodes.get(node2);
        this.imageFiles = imageFiles;

        this.startI = startI;
        this.endI = endI;
        this.startJ = startJ;
        this.endJ = endJ;

        this.level = level;

        this.h = h;
        this.w = w;

        this.correlations = new Correlations();
        this.nrReceivedCorrelations = 0;
        this.nrCorrelationsToReceive = 0;
    }

    CorrelationsActivity setParent(ActivityIdentifier aid) {
        this.parent = aid;
        return this;
    }
}
