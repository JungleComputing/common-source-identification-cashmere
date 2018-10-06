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
import ibis.cashmere.constellation.KernelLaunch;
import ibis.constellation.Timer;

class FastNoiseStage extends Stage {

    public static void executeMC(Device device, int h, int w, String executor, KernelLaunch fn1KL, KernelLaunch fn2KL,
            ExecutorData data) throws CashmereNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "fastnoise");
        int event = timer.start();

        MCL.launchFastnoise1Kernel(fn1KL, h, w, data.temp2, false, data.noisePattern, false);
        MCL.launchFastnoise2Kernel(fn2KL, h, w, data.noisePattern, false, data.temp2, false);
        timer.stop(event);
    }
}
