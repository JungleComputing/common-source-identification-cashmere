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

import java.util.Hashtable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Buffer;

/**
 * The NoisePatternCache is a cache for noise patterns that allows threads to lock a specific noise pattern. The NoisePatternCache
 * keeps track of three types of noise patterns in different memories: - the actual noise patterns, in time domain, kept in the
 * main memory of the node, - noise patterns in the frequency domain, not flipped (regular), kept in the memory of the many-core
 * device - noise patterns in the frequency domain, flipped, kept in the memory of the many-core device
 *
 * The class is thread safe, but the locking is performed on the granularity of a noise pattern. Each noise pattern has an index,
 * and we can lock such an index. We implemented this with a mapping from index to locks for all three variants of noise patterns.
 * The locks are reentrant read/write locks, to make sure that multiple threads can read the noise pattern, but only one can write
 * it.
 */
class NoisePatternCache {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.NoisePatternCache");

    // A mapping from index of the noise pattern to a read/write lock
    static class LockMap extends Hashtable<Integer, ReentrantReadWriteLock> {
        private static final long serialVersionUID = 1L;
    };

    // the actual noise patterns, in time domain, represented by Buffer objects, kept in the main memory
    private static Cache<Buffer> noisePatterns;

    // locks for the noise patterns
    private static LockMap noisePatternLocks;

    // some constants to make the code more readable
    private static final boolean READ = false;
    private static final boolean WRITE = true;
    private static final boolean EVICT = true;
    private static final boolean NO_EVICT = false;
    private static final boolean TIME_DOMAIN_NOISE_PATTERN = true;

    // package private methods

    // initializing

    /**
     * Initialize the noise pattern cache for noise patterns in time and frequency domain.
     *
     * @param height
     *            the height of the noise patterns
     * @param width
     *            the width of the noise patterns
     * @param nrNoisePatterns
     *            the number of noise patterns in time domain
     */
    static void initialize(int height, int width, int nrNoisePatterns) {

        noisePatterns = new Cache<Buffer>("memory");
        noisePatternLocks = new LockMap();

        Buffer[] noisePatternMemory = createNoisePatternMemory(height, width, nrNoisePatterns);

        noisePatterns.setMemory(noisePatternMemory);

        logger.info("Setting # noise patterns in memory to: " + nrNoisePatterns);
    }

    /**
     * Tries to lock noise pattern index. If the noise pattern is already in the cache, we obtain a read lock. If the noise
     * pattern is not in the cache, we obtain a write lock, unless another thread has the write lock. In this case we get a
     * <code>LockException</code>. If we obtain the write lock, we should produce the noise pattern and put it in the data that
     * <code>LockToken</code> points to.
     *
     * @param index
     *            the noise pattern to lock
     * @return a lock token that indicates whether it is a read or write lock and where to put the noise pattern
     * @exception LockException
     *                if the lock cannot be obtained
     */
    static LockToken<Buffer> lockNoisePattern(int index) throws LockException {
        if (logger.isDebugEnabled()) {
            logger.debug("Trying to lock noise pattern {}", index);
        }
        return lock(index, TIME_DOMAIN_NOISE_PATTERN, noisePatternLocks, noisePatterns);
    }

    /**
     * Unlock noise pattern index.
     *
     * @param index
     *            the noise pattern to unlock
     */
    static void unlockNoisePattern(int index) {
        if (logger.isDebugEnabled()) {
            logger.debug("Unlocking noise pattern {}", index);
        }
        unlock(index, TIME_DOMAIN_NOISE_PATTERN, noisePatternLocks, noisePatterns, NO_EVICT);
    }

    /**
     * Clear the cache.
     *
     */
    static void clear() {
    }

    // private methods

    private static Buffer[] createNoisePatternMemory(int height, int width, int nrNoisePatterns) {
        int sizeNoisePattern = height * width * 4;

        Buffer[] noisePatterns = new Buffer[nrNoisePatterns];

        for (int i = 0; i < noisePatterns.length; i++) {
            noisePatterns[i] = new Buffer(sizeNoisePattern);
        }

        return noisePatterns;
    }

    // generic methods

    /*
     * Lock the noise pattern.  If it is in the cache, then we return a read lock, otherwise a write lock.  We briefly lock the
     * whole cache, but this should not be a problem.
     */
    private static <T> LockToken<T> lock(int index, boolean flipped, LockMap locks, Cache<T> cache) throws LockException {
        try {
            synchronized (cache) {
                if (cache.contains(index)) {
                    return lockForRead(index, flipped, locks, cache);
                } else {
                    return lockForWrite(index, flipped, locks, cache);
                }
            }
        } catch (LockException le) {
            if (!le.write) {
                try {
                    Thread.sleep(10);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }
            throw le;
        }
    }

    /*
     * Try to grab a read lock. Since we hold the lock to cache (only briefly), we can try to grab the lock and set the LockToken
     * up without any issues.  We notify to the cache that we locked an item, after which the cache can evict a victim and record
     * that this a recently used noise pattern.  From the cache, we also get a pointer to memory where we can find the noise
     * pattern.
     *
     * pre:
     * - cache is locked
     * - index flipped is in the cache
     * post:
     * - success:
     *   - current thread holds a read lock for index
     *   - LockToken is returned with element index flipped
     *   - index flipped is marked as locked in cache
     * - failure:
     *   - current thread does not hold a read lock
     *   - LockException is thrown
     */
    private static <T> LockToken<T> lockForRead(int index, boolean flipped, LockMap locks, Cache<T> cache) throws LockException {
        String noisePattern = "";
        if (logger.isDebugEnabled()) {
            if (cache == noisePatterns) {
                noisePattern = "noise pattern";
            } else {
                noisePattern = "noise pattern freq " + string(flipped);
            }
        }

        ReentrantReadWriteLock lock = getLock(index, locks);
        if (!lock.readLock().tryLock()) {
            if (logger.isDebugEnabled()) {
                logger.debug(String.format("%s %d in cache, failed to lock for read", noisePattern, index));
            }
            throw new LockException(READ);
        }

        if (logger.isDebugEnabled()) {
            logger.debug(String.format("%s %d in cache, locked for read", noisePattern, index));
        }

        cache.markLocked(index);
        LockToken<T> lt = new LockToken<T>(index, flipped, READ);
        lt.availableElement = cache.get(index);

        return lt;
    }

    /*
     * Helper method to obtain a lock.  If the lock exists, we return it, otherwise we create one.
     *
     * pre:
     * - the cache belonging with this lock is locked
     * post:
     * - locks contains a lock for index and it is returned
     */
    private static ReentrantReadWriteLock getLock(int index, LockMap locks) {
        ReentrantReadWriteLock lock = locks.get(index);
        if (lock == null) {
            lock = new ReentrantReadWriteLock();
            locks.put(index, lock);
        }
        return lock;
    }

    /*
     * Try to obtain a write lock.  From the cache, we try to find an eviction candidate.  The cache may return that there is no
     * victim.  We then to lock the locks of the index we want and possibly the victim.  We do this in a predefined order to
     * ensure that there are no deadlocks possible.  When we successfully locked the index and victim, we notify the cache that
     * we locked an item, we evict the victim and request from the cache which element we can put the noise pattern.
     *
     * pre:
     * - cache is locked
     * - index is not in cache
     * post:
     * - success:
     *   - current thread holds the write lock for index flipped
     *   - LockToken is returned with element index flipped
     *   - if necessary elements are evicted
     *   - index flipped is marked as locked in cache
     * - failure:
     *   - current thread does not hold the write lock
     *   - nothing is evicted
     *   - an exception is thrown
     */
    private static <T> LockToken<T> lockForWrite(int index, boolean flipped, LockMap locks, Cache<T> cache) throws LockException {
        LockToken<T> victims = new LockToken<T>(index, cache.findEvictionCandidate(), flipped, WRITE);

        if (logger.isDebugEnabled()) {
            String noisePattern;
            String message = "";
            if (cache == noisePatterns) {
                noisePattern = "noise pattern";
            } else {
                noisePattern = "noise pattern freq " + string(flipped);
            }

            if (victims.victim != -1) {
                message = String.format(", for evicting %s %d", noisePattern, victims.victim);
            }
            logger.debug(String.format("Trying to lock %s %d for write%s", noisePattern, index, message));
        }

        try {
            lock(victims, locks);
        } catch (LockException e) {
            if (logger.isDebugEnabled()) {
                logger.debug("failed to lock");
            }
            throw e;
        }
        cache.markLocked(index);
        evict(victims, locks, cache);
        victims.availableElement = cache.getAvailableElement(index);
        if (logger.isDebugEnabled()) {
            logger.debug("succeeded to lock");
        }
        return victims;
    }

    /*
     * Locks the locks of the item and the victim in a predefined order to circumvent deadlocks.  The unlocking happens in the
     * reversed order.
     *
     * pre:
     * - the cache belonging to locks is locked
     * post:
     * - success:
     *   - the current thread holds a write lock for lt.index
     *   - the current thread holds a write lock for lt.victim if necessary
     * - failure
     *   - the current thread does not hold a write lock for lt.index
     *   - the current thread does not hold a write lock for lt.victim
     */
    private static void lock(LockToken<?> lt, LockMap locks) throws LockException {
        ReentrantReadWriteLock lock1 = null;
        ReentrantReadWriteLock lock2 = null;

        try {
            // locking means least first
            if (lt.index < lt.victim) {
                if (lt.index != -1) {
                    lock1 = lock(lt.index, locks);
                }
                if (lt.victim != -1) {
                    lock2 = lock(lt.victim, locks);
                }
            } else {
                if (lt.victim != -1) {
                    lock1 = lock(lt.victim, locks);
                }
                if (lt.index != -1) {
                    lock2 = lock(lt.index, locks);
                }
            }
        } catch (LockException e) {
            if (lt.index < lt.victim) {
                tryUnlock(lock2, lt.victim);
                tryUnlock(lock1, lt.index);
            } else {
                tryUnlock(lock2, lt.index);
                tryUnlock(lock1, lt.victim);
            }

            throw e;
        }
    }

    /*
     * pre:
     * - the cache belonging to locks is locked
     * post:
     * - success:
     *   - the current thread holds the write lock index
     *   - the lock is returned
     * - failure:
     *   - the current thread does not hold the write lock
     *   - a lockException is thrown
     */
    private static ReentrantReadWriteLock lock(int index, LockMap locks) throws LockException {

        ReentrantReadWriteLock lock = getLock(index, locks);
        try {
            if (lock.writeLock().tryLock(10, TimeUnit.MILLISECONDS)) {
                return lock;
            } else {
                throw new LockException();
            }
        } catch (InterruptedException e) {
            throw new LockException();
        }
    }

    private static void tryUnlock(ReentrantReadWriteLock lock, int index) {
        if (lock != null && lock.writeLock().isHeldByCurrentThread()) {
            // if (logger.isDebugEnabled()) {
            // logger.debug("unlocking {} for write", index);
            // }
            lock.writeLock().unlock();
        }
    }

    /*
     * Evict a victim and release the write lock to this victim.
     *
     * pre:
     * - cache is locked
     * - the current thread holds a write lock for lt.victim
     * post:
     * - lt.victim is evicted from cache
     * - the write lock for lt.victim is released
     */
    private static <T> void evict(LockToken<?> lt, LockMap locks, Cache<T> cache) {
        // whether it is a noise pattern or a noise pattern freq
        boolean freq = cache != noisePatterns;

        if (lt.victim != -1) {
            cache.evict(lt.victim);
            ReentrantReadWriteLock lock = locks.get(lt.victim);
            lock.writeLock().unlock();

            if (logger.isDebugEnabled()) {
                if (freq) {
                    logger.debug("noise pattern freq {} {} evicted from device", lt.flipped ? "flipped" : "regular", lt.victim);
                } else {
                    logger.debug("noise pattern {} evicted from memory", lt.victim);
                }
            }
        }
    }

    /*
     * Unlock index flipped.  We may want to evict the item, but anyway the item will be markted from being locked to being a
     * possible eviction candidate.
     */
    private static <T> void unlock(int index, boolean flipped, LockMap locks, Cache<T> cache, boolean evict) {
        synchronized (cache) {
            ReentrantReadWriteLock lock = locks.get(index);
            if (lock.isWriteLockedByCurrentThread()) {
                if (evict) {
                    cache.markFromLockedToVictim(index);
                    cache.evict(index);
                } else {
                    cache.markFromLockedToVictim(index);
                }
                lock.writeLock().unlock();
                if (logger.isDebugEnabled()) {
                    logger.debug("unlocking the write lock");
                }
            } else {
                if (lock.getReadLockCount() == 1) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("unlocking the last read lock");
                    }
                    cache.markFromLockedToVictim(index);
                } else {
                    if (logger.isDebugEnabled()) {
                        logger.debug("unlocking one of the read locks, remains locked");
                    }
                }
                lock.readLock().unlock();
            }
        }
    }

    /*
     * Move a write lock to a read lock.
     *
     * pre:
     * - the current thread holds the write lock for index flipped
     * post:
     * - the current thread holds a read lock for index flipped
     */
    private static <T> void toReadLock(int index, boolean flipped, LockMap locks, Cache<T> cache) {
        ReentrantReadWriteLock lock = locks.get(index);
        lock.readLock().lock();
        lock.writeLock().unlock();
    }

    // some methods for retrieving the right datastructure based on whether it is flipped or regular.

    private static String string(boolean flipped) {
        return flipped ? "flipped" : "regular";
    }
}
