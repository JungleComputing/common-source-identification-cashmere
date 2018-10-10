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

package nl.junglecomputing.common_source_identification.multipleGPUs;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.Device;
import ibis.constellation.util.MemorySizes;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;

public class DeviceInfo implements Cloneable {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.DeviceInfo");

    public static DeviceInfo[] info;

    private final Device device;
    private int threshold;
    private final int deviceNo;
    private int nPatternProviders;
    private int nWorkers;
    private ExecutorData executorData;

    private static DeviceInfo[] workers;
    private static DeviceInfo[] providers;
    private static HashMap<String, Integer> workerMap = new HashMap<String, Integer>();
    private static HashMap<String, Integer> providerMap = new HashMap<String, Integer>();

    public static void initialize(int nWorkers, int nPatternProviders, long memoryToBeReservedPerThread, int h, int w,
            int nrBlocksForReduce) {

        int nrWorkers = nWorkers;
        int nrPatternProviders = nPatternProviders;

        List<Device> tempDevices = Cashmere.getDevices("grayscaleKernel");
        ArrayList<Device> devices = new ArrayList<Device>();

        if (tempDevices.size() == 0) {
            throw new Error("No Manycore devices found");
        }

        if (logger.isInfoEnabled()) {
            for (Device d : devices) {
                logger.info("Found device " + d.toString());
            }
        }

        for (Device d : tempDevices) {
            long mem = d.getMemoryCapacity();
            if (mem < memoryToBeReservedPerThread) {
                logger.warn("Device {} does not have enough memory", d.toString());
            } else {
                devices.add(d);
            }
        }

        devices.sort(new Comparator<Device>() {
            @Override
            public int compare(Device o1, Device o2) {
                long m1 = o1.getMemoryCapacity();
                long m2 = o2.getMemoryCapacity();
                return m1 < m2 ? 1 : m1 == m2 ? 0 : -1;
            }
        });

        info = new DeviceInfo[devices.size()];

        if (nrWorkers < info.length) {
            logger.warn("Not enough workers to use all the devices!");
        }

        int[] patternProviders = new int[info.length];
        long[] availableMem = new long[info.length];
        int[] correlationWorkers = new int[info.length];

        for (int i = 0; i < info.length; i++) {
            Device d = devices.get(i);
            info[i] = new DeviceInfo(d, i);
            availableMem[i] = d.getMemoryCapacity();
        }

        long patternSize = h * w * 2 * 4;

        while (nrWorkers > 0) {
            int highestIndex = 0;
            int high = Integer.MIN_VALUE;

            for (int i = 0; i < info.length; i++) {
                // How many noise patterns would be available per worker, on this device, if we would allocate another worker to this device?
                int val = (int) ((availableMem[i] - (correlationWorkers[i] + 1) * memoryToBeReservedPerThread) / patternSize)
                        / (correlationWorkers[i] + 1);
                if (val > high) {
                    high = val;
                    highestIndex = i;
                }
            }
            correlationWorkers[highestIndex]++;
            nrWorkers--;
        }

        // We may have devices that have nothing allocated to them.
        int from = 0;
        for (int i = info.length - 1; i >= 0; i--) {
            if (correlationWorkers[i] == 0) {
                if (correlationWorkers[from] > 1) {
                    correlationWorkers[i] = 1;
                    correlationWorkers[from++] -= 1;
                } else {
                    patternProviders[i]++;
                    nrPatternProviders--;
                }
            }
        }

        // Now, we still have to divide the pattern providers.
        while (nrPatternProviders > 0) {
            int highestIndex = 0;
            int high = Integer.MIN_VALUE;

            for (int i = 0; i < info.length; i++) {
                // How many noise patterns would be available per worker, on this device, if we would allocate a pattern provider to this device?
                int val;
                if (correlationWorkers[i] == 0) {
                    val = Integer.MAX_VALUE;
                } else {
                    val = (int) ((availableMem[i]
                            - (correlationWorkers[i] + patternProviders[i] + 1) * memoryToBeReservedPerThread) / patternSize)
                            / correlationWorkers[i];
                }
                if (val > high) {
                    high = val;
                    highestIndex = i;
                }
            }
            patternProviders[highestIndex]++;
            nrPatternProviders--;
        }

        for (int i = 0; i < info.length; i++) {
            info[i].setnPatternProviders(patternProviders[i]);
            info[i].setnWorkers(correlationWorkers[i]);
            if (correlationWorkers[i] > 0) {
                int threshold = (int) ((availableMem[i]
                        - (correlationWorkers[i] + patternProviders[i]) * memoryToBeReservedPerThread - 200 * MemorySizes.MB)
                        / patternSize) / correlationWorkers[i];
                if (threshold % 2 != 0) {
                    threshold--;
                }
                info[i].setThreshold(threshold);
            }

            if (logger.isInfoEnabled()) {
                logger.info("Device {}: workers: {}, providers: {}, threshold {}", i, info[i].nWorkers, info[i].nPatternProviders,
                        info[i].threshold);
            }
        }

        workers = new DeviceInfo[nWorkers];
        nrWorkers = 0;
        providers = new DeviceInfo[nPatternProviders];
        nrPatternProviders = 0;
        try {
            for (DeviceInfo d : info) {
                for (int i = 0; i < d.nWorkers; i++) {
                    DeviceInfo newd = (DeviceInfo) d.clone();
                    workers[nrWorkers++] = newd;
                    newd.setExecutorData(new ExecutorData(newd.device, h, w, nrBlocksForReduce, false));
                }
                for (int i = 0; i < d.nPatternProviders; i++) {
                    DeviceInfo newd = (DeviceInfo) d.clone();
                    providers[nrPatternProviders++] = newd;
                    newd.setExecutorData(new ExecutorData(newd.device, h, w, nrBlocksForReduce, false));
                }
            }
        } catch (CloneNotSupportedException e) {
            logger.error("Should not happen!", e);
            throw new Error(e);
        }
    }

    private static DeviceInfo getInfo(String worker, DeviceInfo[] infos, Map<String, Integer> map) {
        int index;
        synchronized (map) {
            int sz = map.size();
            Integer v = map.putIfAbsent(worker, new Integer(sz));
            if (v == null) {
                index = sz;
            } else {
                index = v;
            }
        }
        return infos[index];
    }

    public static DeviceInfo getDeviceInfo(String worker, String task) {
        if (task == CorrelationsActivity.LABEL) {
            return getInfo(worker, workers, workerMap);
        }
        return getInfo(worker, providers, providerMap);
    }

    public DeviceInfo(Device device, int deviceNo) {
        this.device = device;
        this.deviceNo = deviceNo;
    }

    public int getDeviceNo() {
        return deviceNo;
    }

    public Device getDevice() {
        return device;
    }

    public int getThreshold() {
        return threshold;
    }

    private void setThreshold(int threshold) {
        this.threshold = threshold;
    }

    public int getnPatternProviders() {
        return nPatternProviders;
    }

    private void setnPatternProviders(int nPatternProviders) {
        this.nPatternProviders = nPatternProviders;
    }

    public int getnWorkers() {
        return nWorkers;
    }

    private void setnWorkers(int nWorkers) {
        this.nWorkers = nWorkers;
    }

    public ExecutorData getExecutorData() {
        return executorData;
    }

    private void setExecutorData(ExecutorData executorData) {
        this.executorData = executorData;
    }

}
