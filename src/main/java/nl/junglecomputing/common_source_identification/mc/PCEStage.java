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

import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;

import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.Kernel;
import ibis.cashmere.constellation.KernelLaunch;
import ibis.cashmere.constellation.LibFunc;
import ibis.cashmere.constellation.LibFuncLaunch;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Timer;
import nl.junglecomputing.common_source_identification.cpu.Stage;

/*
 * In this stage we compute the Peak-To-Correlation Energy of two noise patterns in the time domain
 */

class PCEStage extends Stage {

    static final int SQUARE_SIZE = 11;

    static double executeMC(Device device, Pointer x, Pointer y, int h, int w, String executor, int nrBlocksForReduce,
            ExecutorData data) throws CashmereNotAvailable, LibFuncNotAvailable {

        Timer timer = Cashmere.getTimer("MC", executor, "pce");
        int tevent = timer.start();

        float[] peak = new float[1];
        int nrBlocksFindPeak = nrBlocksForReduce;

        int nrBlocksEnergy = nrBlocksForReduce;
        double[] energy = new double[1];

        Kernel ccKernel = Cashmere.getKernel("crossCorrelateKernel", device);

        KernelLaunch ccKL = ccKernel.createLaunch();

        LibFunc fft = Cashmere.getLibFunc("fft", device);

        LibFuncLaunch ifftLaunch = fft.createLaunch();

        Kernel fpKernel = Cashmere.getKernel("findPeakKernel", device);

        KernelLaunch fpKL = fpKernel.createLaunch();

        Kernel mlfKernel = Cashmere.getKernel("maxLocFloatsKernel", device);

        KernelLaunch mlfKL = mlfKernel.createLaunch();

        Kernel ceKernel = Cashmere.getKernel("computeEnergyKernel", device);

        KernelLaunch ceKL = ceKernel.createLaunch();

        Kernel sdKernel = Cashmere.getKernel("sumDoublesKernel", device);

        KernelLaunch sdKL = sdKernel.createLaunch();
        MCL.launchCrossCorrelateKernel(ccKL, h * w, data.crossCorrelation, false, x, false, y, false); // 3.6ms

        ifftLaunch.launch((cl_command_queue queue, int num_events_in_wait_list, cl_event[] event_wait_list, cl_event event) -> FFT
                .performFFT(queue, h, w, data.crossCorrelation, data.temp, false, num_events_in_wait_list, event_wait_list,
                        event));

        MCL.launchFindPeakKernel(fpKL, nrBlocksFindPeak, h * w, data.peak, false, data.tempPeaks, false, data.indicesPeak, false,
                data.crossCorrelation, false); // 5.2 ms
        MCL.launchMaxLocFloatsKernel(mlfKL, nrBlocksFindPeak, data.dummyPeak, false, data.indexPeak, false, data.tempPeaks, false,
                data.indicesPeak, false); // 5.2 ms
        MCL.launchComputeEnergyKernel(ceKL, nrBlocksEnergy, h, w, data.tempEnergy, false, data.indexPeak, false,
                data.crossCorrelation, false); // 6.6 ms
        MCL.launchSumDoublesKernel(sdKL, nrBlocksEnergy, data.energy, false, data.tempEnergy, false);

        device.get(energy, data.energy);
        device.get(peak, data.peak);

        double absPce = ((double) peak[0] * (double) peak[0]) / energy[0];
        timer.stop(tevent);

        return absPce;
    }
}
