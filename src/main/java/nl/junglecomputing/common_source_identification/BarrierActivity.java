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
 * Activity that acts as a barrier. It interacts with NotifyParentActivity that will set the parent for node activities.  This
 * will be the sign that we can send a message back to the parent.
 */
class BarrierActivity extends Activity {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.BarrierActivity");

    static final String LABEL = "BarrierActivity";
    static volatile ActivityIdentifier parent = null;

    BarrierActivity(String node) {
        super(new Context(node + BarrierActivity.LABEL), true, false);
    }

    /* 
     * Waits until parent is set and then finishes.
     */
    @Override
    public int initialize(Constellation constellation) {
        logger.debug("Trying to lock the BarrierActivity");
        synchronized (LABEL) {
            logger.debug("succeeded to lock");
            while (parent == null) {
                logger.debug("waiting for the parent to be set");
                try {
                    LABEL.wait();
                } catch (InterruptedException e) {
                    throw new Error(e);
                }
            }
            logger.debug("Parent activity has been set");
            return FINISH;
        }
    }

    @Override
    public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        if (parent == null) {
            throw new Error("Should not happen");
        }
        logger.debug("Sending an event to the parent");
        constellation.send(new Event(identifier(), parent, null));
    }
}
