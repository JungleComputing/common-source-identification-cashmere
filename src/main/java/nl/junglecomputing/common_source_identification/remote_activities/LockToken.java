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

class LockToken<T> {
    final int index;
    final int victim;
    final boolean flipped;
    T availableElement;

    private final boolean writeLock;

    LockToken(int index, boolean flipped, boolean writeLock) {
        this(index, -1, flipped, writeLock);
    }

    LockToken(int index, int victim, boolean flipped, boolean writeLock) {
        this.index = index;
        this.victim = victim;
        this.flipped = flipped;
        this.writeLock = writeLock;
    }

    boolean readLock() {
        return !writeLock;
    }

    boolean writeLock() {
        return writeLock;
    }
}
