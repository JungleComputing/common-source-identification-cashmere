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

package nl.junglecomputing.common_source_identification.main_mem_cache;

/*
 * An exception that indicates that we failed to lock a lock.  It also indicates whether a read or write lock failed.
 */
public class LockException extends Exception {

    private static final long serialVersionUID = 1L;

    public boolean write;

    public LockException(boolean write) {
        this.write = write;
    }

    public LockException() {
        this(true);
    }
}
