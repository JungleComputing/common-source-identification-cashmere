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

package nl.junglecomputing.common_source_identification;

import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;

import ibis.cashmere.constellation.Argument;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.LibFunc;
import ibis.cashmere.constellation.LibFuncLaunch;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Timer;

class ComputeFrequencyDomain {

    static void computeFreq(Device device, Pointer noisePatternFreq, int h, int w, boolean flipped, String executor,
            ExecutorData data) throws CashmereNotAvailable, LibFuncNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "computeFreq");
        int tevent = timer.start();

        Kernel tcKernel = Cashmere.getKernel(flipped ? "toComplexAndFlipKernel" : "toComplexKernel", device);
        KernelLaunch tcKL = tcKernel.createLaunch();

        LibFunc fft = Cashmere.getLibFunc("fft", device);
        LibFuncLaunch fftLaunch = fft.createLaunch();

        fftLaunch.setArgument(h, Argument.Direction.IN);
        fftLaunch.setArgument(w, Argument.Direction.IN);
        fftLaunch.setArgumentNoCopy(noisePatternFreq, Argument.Direction.INOUT);
        fftLaunch.setArgumentNoCopy(data.temp2, Argument.Direction.INOUT);

        if (flipped) {
            MCL.launchToComplexAndFlipKernel(tcKL, h, w, noisePatternFreq, false, data.noisePattern, false);
        } else {
            MCL.launchToComplexKernel(tcKL, h * w, noisePatternFreq, false, data.noisePattern, false);
        }

        fftLaunch.launch((cl_command_queue queue, int numEventsWaitList, cl_event[] event_wait_list, cl_event event) -> FFT
                .performFFT(queue, h, w, noisePatternFreq, data.temp2, true, numEventsWaitList, event_wait_list, event));

        timer.stop(tevent);
    }
}
