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

import java.nio.ByteBuffer;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.constellation.Timer;

import nl.junglecomputing.common_source_identification.cpu.Stage;

class ImageToGrayscaleStage extends Stage {

    static void executeMC(Device device, Buffer image, int h, int w, String executor, ExecutorData data)
            throws CashmereNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "convert to grayscale");
        int event = timer.start();

        Kernel kernel = Cashmere.getKernel("grayscaleKernel", device);

        device.copy(image, data.imagePointer);

        if (logger.isDebugEnabled()) {
            logger.debug("Doing image to grayscale stage");
        }

        KernelLaunch kernelLaunch = kernel.createLaunch();

        MCL.launchGrayscaleKernel(kernelLaunch, h * w, data.noisePattern, false, data.imagePointer, false);
        timer.stop(event);

    }
}
