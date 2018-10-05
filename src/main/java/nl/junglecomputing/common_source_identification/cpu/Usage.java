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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class Usage {

    public static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.Usage");

    private Nodes locked;
    private Nodes victims;
    private int maxNrItems;

    private String description;

    Usage(int maxNrItems, String description) {
        initialize();
        setMaxNrItems(maxNrItems);
        this.description = description;
    }

    void initialize() {
        locked = new Nodes();
        victims = new Nodes();
    }

    void setMaxNrItems(int maxNrItems) {
        this.maxNrItems = maxNrItems;
        if (logger.isDebugEnabled()) {
            logger.debug("{}: Setting the cache to {} items", description, maxNrItems);
        }
    }

    void markLocked(int index) {
        Node node = victims.remove(index);
        if (node == null) {
            locked.addToHead(index);
        } else {
            locked.addToHead(node);
        }
        if (logger.isDebugEnabled()) {
            logger.debug("{}: marking {} locked", description, index);
            // logger.debug("{}: locked: {}", description, locked);
            // logger.debug("{}: victims: {}", description, victims);
        }
    }

    void markFromLockedToVictim(int index) {
        Node node = locked.remove(index);
        victims.addToHead(node);
        if (logger.isDebugEnabled()) {
            logger.debug("{}: marking {} from locked to victim", description, index);
            // logger.debug("{}: locked: {}", description, locked);
            // logger.debug("{}: victims: {}", description, victims);
        }
    }

    int findEvictionCandidate() {
        if (maximumAchieved()) {
            int index = victims.getTailIndex();
            if (logger.isDebugEnabled()) {
                logger.debug("{}: Candidate for evicting: {}", description, index);
                // logger.debug("{}: victims: {}", description, victims);
                // logger.debug("{}: {} items in cache", description, nodes.size());
            }
            return index;
        }
        return -1;
    }

    void evict(int index) {
        victims.remove(index);
        // if (logger.isDebugEnabled()) {
        // logger.debug("{}: evicted {}", description, index);
        // logger.debug("{}: victims: {}", description, victims);
        // }
    }

    private boolean maximumAchieved() {
        if (logger.isDebugEnabled()) {
            logger.debug("{}: find a candidate for eviction", description);
            logger.debug("{}: victims.size: {}", description, victims.size());
            logger.debug("{}: locked.size: {}", description, locked.size());
        }
        return victims.size() + locked.size() >= maxNrItems;
    }
}
