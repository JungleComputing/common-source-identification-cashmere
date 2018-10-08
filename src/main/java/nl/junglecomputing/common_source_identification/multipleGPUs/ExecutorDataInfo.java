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
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Device;
import ibis.constellation.Constellation;
import nl.junglecomputing.common_source_identification.mc.ExecutorData;

/**
 * ExecutorData for each executor, each device.
 */
public class ExecutorDataInfo {

    private static ArrayList<Map<Constellation, ExecutorData>> executorDataMaps = new ArrayList<Map<Constellation, ExecutorData>>();
    private static ArrayList<ArrayList<ExecutorData>> nonUsedExecutorData = new ArrayList<ArrayList<ExecutorData>>();
    private static List<Device> devices;

    /*
     * At initialization time we don't know yet to which executors this data will mapped, therefore, we store it in a list and
     * defer creating the mapping to a later stage.
     */
    public static synchronized void initialize(int nrExecutors, List<Device> devices, int h, int w, int nrBlocksForReduce) {
        ExecutorDataInfo.devices = devices;
        for (int deviceNo = 0; deviceNo < devices.size(); deviceNo++) {
            executorDataMaps.add(new IdentityHashMap<Constellation, ExecutorData>());
            nonUsedExecutorData.add(new ArrayList<ExecutorData>());
        }
        for (int i = 0; i < nrExecutors; i++) {
            Buffer bufferHWRGB = new Buffer(h * w * 3);
            for (Device device : devices) {
                int deviceNo = getDeviceNo(device);
                nonUsedExecutorData.get(deviceNo).add(new ExecutorData(device, bufferHWRGB, h, w, nrBlocksForReduce, false));
            }
        }
    }

    private static int getDeviceNo(Device d) {
        for (int i = 0; i < devices.size(); i++) {
            if (devices.get(i) == d) {
                return i;
            }
        }
        throw new Error("Device not found: " + d.toString());
    }

    /*
     * Get the executor data with a specific executor.  If an executor does not yet have data, we will assign it data.
     */
    public static synchronized ExecutorData get(Device device, Constellation executor) {
        int deviceNo = getDeviceNo(device);
        ExecutorData executorData = executorDataMaps.get(deviceNo).get(executor);
        if (executorData == null) {
            executorData = nonUsedExecutorData.get(deviceNo).remove(0);
            executorDataMaps.get(deviceNo).put(executor, executorData);
        }
        return executorData;
    }
}
