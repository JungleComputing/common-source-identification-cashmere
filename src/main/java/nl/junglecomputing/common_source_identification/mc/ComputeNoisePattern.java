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

import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import nl.junglecomputing.common_source_identification.cpu.FileToImageStage;

public class ComputeNoisePattern {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.ComputeNoisePattern");

    public static AtomicInteger patternsComputed = new AtomicInteger(0);

    public static void computePRNU_MC(int index, File file, int h, int w, String executor, Device device, ExecutorData data)
            throws IOException, CashmereNotAvailable, LibFuncNotAvailable {

        if (logger.isDebugEnabled()) {
            logger.debug("Computing PRNU for {} on mc", index);
        }

        Buffer image = getImage(file.getPath(), h, w, executor, data);

        Kernel fn1Kernel = Cashmere.getKernel("fastnoise1Kernel", device);
        Kernel fn2Kernel = Cashmere.getKernel("fastnoise2Kernel", device);

        KernelLaunch fn1KL = fn1Kernel.createLaunch();
        KernelLaunch fn2KL = fn2Kernel.createLaunch();

        ImageToGrayscaleStage.executeMC(device, image, h, w, executor, data);

        FastNoiseStage.executeMC(device, h, w, executor, fn1KL, fn2KL, data);

        ZeroMeanStage.executeMC(device, h, w, executor, data);

        WienerStage.executeMC(device, h, w, executor, data);

        patternsComputed.incrementAndGet();
    }

    static Buffer getImage(String filename, int h, int w, String executor, ExecutorData data) throws IOException {
        return FileToImageStage.execute(filename, h, w, executor, data.imageBuffer);
    }
}
