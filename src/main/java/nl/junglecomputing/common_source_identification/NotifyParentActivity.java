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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;

/*
 * Notify nodes of which node is the parent and progress activity.
 */
class NotifyParentActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.NotifyParentActivity");

    static final String LABEL = "NotifyParentActivity";

    private ActivityIdentifier parent = null;
    private ActivityIdentifier progressActivityID = null;

    NotifyParentActivity(String node, ActivityIdentifier parent, ActivityIdentifier progressActivityID) {
        super(new Context(node + NotifyParentActivity.LABEL), true, false);

        this.parent = parent;
        this.progressActivityID = progressActivityID;
    }

    /*
     * Set the parent of the BarrierActivity, and notify it so it can report back that the barrier has been taken.
     */
    @Override
    public int initialize(Constellation cons) {
        logger.debug("Trying to lock the BarrierActivity");
        synchronized (BarrierActivity.LABEL) {
            logger.debug("succeeded to lock");
            logger.debug("Setting the parent activity");
            BarrierActivity.parent = parent;
            LeafCorrelationsActivity.progressActivityID = progressActivityID;
            logger.debug("Notifying the BarrierActivity");
            BarrierActivity.LABEL.notifyAll();
        }
        return FINISH;
    }

    @Override
    public int process(Constellation cons, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation cons) {
    }
}
