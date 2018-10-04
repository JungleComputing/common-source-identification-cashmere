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

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 * This class represents a Cache with an array of elements of type T that represents the memory.  The memory is accessed by means
 * of indices, that we will name memoryIndices.  The items that go into this cache are assumed to have indices as well and we are
 * going to call these indices itemIndices.  We have a concurrent hash map that maps the itemIndices to the memoryIndices.  The
 * cache keeps track of the amount of available slots in the memory with a list of memoryIndices.  There is also a Usage object
 * that implements the eviction strategy.
 */
class Cache<T> {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.Cache");

    private ConcurrentHashMap<Integer, Integer> memoryToItemIndices;
    private ArrayList<Integer> availableMemoryIndices;

    // usage keeps track of which items must be evicted
    private Usage usage;
    private T[] memory;

    Cache(String description) {
        memoryToItemIndices = new ConcurrentHashMap<Integer, Integer>();
        usage = new Usage(10, description);
    }

    // Set the memory and inform Usage.
    void setMemory(T[] memory) {
        this.memory = memory;
        usage.setMaxNrItems(memory.length);
        availableMemoryIndices = new ArrayList<Integer>(memory.length);
        for (int i = 0; i < memory.length; i++) {
            availableMemoryIndices.add(i, i);
        }
    }

    // Get an item from the cache based on an itemIndex.
    T get(int index) {
        return memory[memoryToItemIndices.get(index)];
    }

    // mark an item as locked
    void markLocked(int index) {
        usage.markLocked(index);
    }

    int findEvictionCandidate() {
        return usage.findEvictionCandidate();
    }

    // get an element from the cache that is available
    T getAvailableElement(int index) {
        int indexMemory = availableMemoryIndices.remove(0);
        memoryToItemIndices.put(index, indexMemory);
        return memory[indexMemory];
    }

    // mark a locked item to being a victim
    void markFromLockedToVictim(int index) {
        usage.markFromLockedToVictim(index);
    }

    // evict an item, add it to the available memory indices
    T evict(int index) {
        usage.evict(index);

        int indexMemory;
        indexMemory = memoryToItemIndices.remove(index);
        availableMemoryIndices.add(0, indexMemory);

        return memory[indexMemory];
    }

    // whether item with index index is in the cache
    boolean contains(int index) {
        return memoryToItemIndices.containsKey(index);
    }

    // retrieve an enumeration of all indices
    synchronized Enumeration<Integer> indices() {
        return memoryToItemIndices.keys();
    }

    // clear the cache
    synchronized void clear() {
        memoryToItemIndices.clear();
        usage.initialize();
    }
}
