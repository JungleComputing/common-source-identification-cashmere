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

package nl.junglecomputing.common_source_identification.mc;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.constellation.Timer;

import nl.junglecomputing.common_source_identification.cpu.Stage;

/*
 * Stage that applies the ZeroMean filter
 */
public class ZeroMeanStage extends Stage {

    public static void executeMC(Device device, int h, int w, String executor, ExecutorData data) throws CashmereNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "zeromean");
        int event = timer.start();

        Kernel verKernel = Cashmere.getKernel("zeromeanVerticallyKernel", device);
        Kernel traKernel = Cashmere.getKernel("transposeKernel", device);

        KernelLaunch ver1KL = verKernel.createLaunch();
        KernelLaunch tra1KL = traKernel.createLaunch();
        KernelLaunch ver2KL = verKernel.createLaunch();
        KernelLaunch tra2KL = traKernel.createLaunch();

        if (logger.isDebugEnabled()) {
            logger.debug("Executing the zeromean stage");
        }

        MCL.launchZeromeanVerticallyKernel(ver1KL, h, w, data.temp2, false, data.noisePattern, false);
        MCL.launchTransposeKernel(tra1KL, w, h, data.noisePattern, false, data.temp2, false);
        MCL.launchZeromeanVerticallyKernel(ver2KL, w, h, data.temp2, false, data.noisePattern, false);
        MCL.launchTransposeKernel(tra2KL, h, w, data.noisePattern, false, data.temp2, false);

        timer.stop(event);
    }
}
