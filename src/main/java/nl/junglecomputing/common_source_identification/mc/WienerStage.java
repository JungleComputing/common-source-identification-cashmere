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
import ibis.cashmere.constellation.CommandStream;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.DeviceEvent;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.LibFunc;
import ibis.cashmere.constellation.LibFuncLaunch;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Timer;

import nl.junglecomputing.common_source_identification.cpu.Stage;

/*
 * Stage that applies the Wiener filter.
 */
public class WienerStage extends Stage {

    public static void executeMC(Device device, int h, int w, String executor, ExecutorData data)
            throws CashmereNotAvailable, LibFuncNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "wiener");

        int tevent = timer.start();

        // convert input to complex values
        Kernel comKernel = Cashmere.getKernel("toComplexKernel", device);

        KernelLaunch comKL = comKernel.createLaunch();

        if (logger.isDebugEnabled()) {
            logger.debug("Executing the Wiener stage");
        }

        MCL.launchToComplexKernel(comKL, h * w, data.complex, false, data.noisePattern, false);

        Kernel vzmKernel = Cashmere.getKernel("varianceZeroMeanKernel", device);

        KernelLaunch vzmKL = vzmKernel.createLaunch();

        MCL.launchVarianceZeroMeanKernel(vzmKL, h * w, data.variance, false, data.noisePattern, false);

        // forward fft
        LibFunc fft = Cashmere.getLibFunc("fft", device);
        LibFuncLaunch fftLaunch = fft.createLaunch();
        fftLaunch.launch((CommandStream queue, DeviceEvent[] event_wait_list) -> FFT
                .performFFT(queue, h, w, data.complex, data.temp2, true, event_wait_list));

        Kernel sqMKernel = Cashmere.getKernel("computeSquaredMagnitudesKernel", device);
        KernelLaunch sqKL = sqMKernel.createLaunch();

        MCL.launchComputeSquaredMagnitudesKernel(sqKL, h, w, data.noisePattern, false, data.complex, false);

        Kernel veKernel = Cashmere.getKernel("computeVarianceEstimatesKernel", device);

        KernelLaunch veKL = veKernel.createLaunch();

        MCL.launchComputeVarianceEstimatesKernel(veKL, h, w, data.tempFloat, false, data.noisePattern, false);

        // scale the frequencies with the global and local variance
        Kernel fsKernel = Cashmere.getKernel("scaleWithVariancesKernel", device);

        KernelLaunch fsKL = fsKernel.createLaunch();

        MCL.launchScaleWithVariancesKernel(fsKL, h, w, data.complex, false, data.tempFloat, false, data.variance, false);

        // inverse Fourier transform
        LibFuncLaunch ifftLaunch = fft.createLaunch();
        ifftLaunch.launch((CommandStream queue, DeviceEvent[] event_wait_list) -> FFT
                .performFFT(queue, h, w, data.complex, data.temp2, false, event_wait_list));

        // convert values to real and return result

        Kernel trKernel = Cashmere.getKernel("toRealKernel", device);

        KernelLaunch trKL = trKernel.createLaunch();

        MCL.launchToRealKernel(trKL, h * w, data.noisePattern, false, data.complex, false);

        timer.stop(tevent);
    }
}
