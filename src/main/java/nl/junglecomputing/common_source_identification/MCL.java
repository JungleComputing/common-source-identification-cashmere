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

import ibis.cashmere.constellation.Argument;
import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.KernelLaunch;

class MCL {

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, byte[] input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, byte[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, Buffer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, Pointer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, byte[] input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, byte[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, Buffer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, Pointer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, byte[] input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, byte[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, Buffer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, Pointer input) throws CashmereNotAvailable {
        launchGrayscaleKernel(kl, n, output, true, input, true);
    }

    static void launchGrayscaleKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, float[] input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, boolean copydxsdys, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, Buffer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, boolean copydxsdys, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, Pointer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, float[] dxsdys, boolean copydxsdys, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, float[] input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, boolean copydxsdys, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, Buffer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, boolean copydxsdys, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, Pointer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Buffer dxsdys, boolean copydxsdys, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, float[] input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, boolean copydxsdys, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, Buffer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, boolean copydxsdys, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, Pointer input) throws CashmereNotAvailable {
        launchFastnoise1Kernel(kl, h, w, dxsdys, true, input, true);
    }

    static void launchFastnoise1Kernel(KernelLaunch kl, int h, int w, Pointer dxsdys, boolean copydxsdys, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, float[] dxsdys)
            throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, float[] dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, Buffer dxsdys) throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Buffer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, Pointer dxsdys)
            throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Pointer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, float[] dxsdys) throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, float[] dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, Buffer dxsdys) throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Buffer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, Pointer dxsdys) throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Pointer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, float[] dxsdys)
            throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, float[] dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, Buffer dxsdys) throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Buffer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, Pointer dxsdys)
            throws CashmereNotAvailable {
        launchFastnoise2Kernel(kl, h, w, output, true, dxsdys, true);
    }

    static void launchFastnoise2Kernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Pointer dxsdys,
            boolean copydxsdys) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copydxsdys) {
            kl.setArgument(dxsdys, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(dxsdys, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, float[] input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, Buffer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, Pointer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, float[] input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, float[] input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer input)
            throws CashmereNotAvailable {
        launchZeromeanVerticallyKernel(kl, h, w, output, true, input, true);
    }

    static void launchZeromeanVerticallyKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * 1, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * 1, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, float[] input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, Buffer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, Pointer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, float[] input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, float[] input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer input) throws CashmereNotAvailable {
        launchTransposeKernel(kl, h, w, output, true, input, true);
    }

    static void launchTransposeKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, float[] input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, Buffer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, Pointer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, float[] input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, Buffer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, Pointer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, float[] input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, Buffer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, Pointer input) throws CashmereNotAvailable {
        launchToComplexKernel(kl, n, output, true, input, true);
    }

    static void launchToComplexKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, float[] input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, Buffer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, Pointer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, float[] input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, float[] input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer input)
            throws CashmereNotAvailable {
        launchComputeSquaredMagnitudesKernel(kl, h, w, output, true, input, true);
    }

    static void launchComputeSquaredMagnitudesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates, float[] input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates,
            boolean copyvarianceEstimates, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates, Buffer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates,
            boolean copyvarianceEstimates, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates, Pointer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, float[] varianceEstimates,
            boolean copyvarianceEstimates, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates, float[] input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates,
            boolean copyvarianceEstimates, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates, Buffer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates,
            boolean copyvarianceEstimates, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates, Pointer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Buffer varianceEstimates,
            boolean copyvarianceEstimates, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates, float[] input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates,
            boolean copyvarianceEstimates, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates, Buffer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates,
            boolean copyvarianceEstimates, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates, Pointer input)
            throws CashmereNotAvailable {
        launchComputeVarianceEstimatesKernel(kl, h, w, varianceEstimates, true, input, true);
    }

    static void launchComputeVarianceEstimatesKernel(KernelLaunch kl, int h, int w, Pointer varianceEstimates,
            boolean copyvarianceEstimates, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, float[] input)
            throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, boolean copyvariance, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, Buffer input) throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, boolean copyvariance, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, Pointer input)
            throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, float[] variance, boolean copyvariance, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, float[] input) throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, boolean copyvariance, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, Buffer input) throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, boolean copyvariance, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, Pointer input) throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Buffer variance, boolean copyvariance, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, float[] input)
            throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, boolean copyvariance, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, Buffer input) throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, boolean copyvariance, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, Pointer input)
            throws CashmereNotAvailable {
        launchVarianceZeroMeanKernel(kl, n, variance, true, input, true);
    }

    static void launchVarianceZeroMeanKernel(KernelLaunch kl, int n, Pointer variance, boolean copyvariance, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 1024;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, float[] varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, float[] varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, float[] varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Buffer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Buffer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Buffer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Pointer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Pointer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, Pointer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, float[] varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, float[] varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, float[] varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, float[] varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, float[] varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, float[] varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            float[] varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Buffer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer varianceEstimates,
            float[] variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, float[] variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer varianceEstimates,
            Buffer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Buffer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer varianceEstimates,
            Pointer variance) throws CashmereNotAvailable {
        launchScaleWithVariancesKernel(kl, h, w, output, true, varianceEstimates, true, variance, true);
    }

    static void launchScaleWithVariancesKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput,
            Pointer varianceEstimates, boolean copyvarianceEstimates, Pointer variance, boolean copyvariance)
            throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.INOUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.INOUT);
        }
        if (copyvarianceEstimates) {
            kl.setArgument(varianceEstimates, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(varianceEstimates, Argument.Direction.IN);
        }
        if (copyvariance) {
            kl.setArgument(variance, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(variance, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, float[] input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, Buffer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, Pointer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, float[] output, boolean copyoutput, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, float[] input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, Buffer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, Pointer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Buffer output, boolean copyoutput, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, float[] input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, Buffer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, Pointer input) throws CashmereNotAvailable {
        launchToRealKernel(kl, n, output, true, input, true);
    }

    static void launchToRealKernel(KernelLaunch kl, int n, Pointer output, boolean copyoutput, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(1024, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, float[] input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, Buffer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, Pointer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, float[] output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, float[] input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, Buffer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, Pointer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Buffer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, float[] input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, float[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, Buffer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, Pointer input)
            throws CashmereNotAvailable {
        launchToComplexAndFlipKernel(kl, h, w, output, true, input, true);
    }

    static void launchToComplexAndFlipKernel(KernelLaunch kl, int h, int w, Pointer output, boolean copyoutput, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyoutput) {
            kl.setArgument(output, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(output, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsW = Math.min(16, w);
            int nrThreadsW = w == 1 * nrVectorsW ? 1
                    : w % (1 * nrVectorsW) == 0 ? w / (1 * nrVectorsW) : w / (1 * nrVectorsW) + 1;
            kl.launch(nrVectorsW * nrThreadsW, 1 * h, 1 * 1, nrVectorsW, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsW = Math.min(1024, w);
            int nrBlocksW = w == 1 * nrThreadsW ? 1 : w % (1 * nrThreadsW) == 0 ? w / (1 * nrThreadsW) : w / (1 * nrThreadsW) + 1;
            int nrThreadsNrThreadsW = Math.min(32, nrThreadsW);
            int nrWarpsNrThreadsW = nrThreadsW == 1 * nrThreadsNrThreadsW ? 1
                    : nrThreadsW % (1 * nrThreadsNrThreadsW) == 0 ? nrThreadsW / (1 * nrThreadsNrThreadsW)
                            : nrThreadsW / (1 * nrThreadsNrThreadsW) + 1;
            kl.launch(nrThreadsNrThreadsW * nrBlocksW, nrWarpsNrThreadsW * h, 1 * 1, nrThreadsNrThreadsW, nrWarpsNrThreadsW, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, float[] x, float[] y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, float[] x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, float[] x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, float[] x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, float[] x, Pointer y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, float[] x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Buffer x, float[] y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Buffer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Buffer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Buffer x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Buffer x, Pointer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Buffer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Pointer x, float[] y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Pointer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Pointer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Pointer x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, Pointer x, Pointer y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, float[] out, boolean copyout, Pointer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, float[] x, float[] y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, float[] x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, float[] x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, float[] x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, float[] x, Pointer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, float[] x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Buffer x, float[] y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Buffer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Buffer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Buffer x, boolean copyx, Buffer y,
            boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Buffer x, Pointer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Buffer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Pointer x, float[] y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Pointer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Pointer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Pointer x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, Pointer x, Pointer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Buffer out, boolean copyout, Pointer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, float[] x, float[] y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, float[] x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, float[] x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, float[] x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, float[] x, Pointer y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, float[] x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Buffer x, float[] y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Buffer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Buffer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Buffer x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Buffer x, Pointer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Buffer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Pointer x, float[] y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Pointer x, boolean copyx,
            float[] y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Pointer x, Buffer y) throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Pointer x, boolean copyx,
            Buffer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, Pointer x, Pointer y)
            throws CashmereNotAvailable {
        launchCrossCorrelateKernel(kl, n, out, true, x, true, y, true);
    }

    static void launchCrossCorrelateKernel(KernelLaunch kl, int n, Pointer out, boolean copyout, Pointer x, boolean copyx,
            Pointer y, boolean copyy) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyout) {
            kl.setArgument(out, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(out, Argument.Direction.OUT);
        }
        if (copyx) {
            kl.setArgument(x, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(x, Argument.Direction.IN);
        }
        if (copyy) {
            kl.setArgument(y, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(y, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreadsN = Math.min(512, n);
            int nrBlocksN = n == 1 * nrThreadsN ? 1 : n % (1 * nrThreadsN) == 0 ? n / (1 * nrThreadsN) : n / (1 * nrThreadsN) + 1;
            int nrThreadsNrThreadsN = Math.min(32, nrThreadsN);
            int nrWarpsNrThreadsN = nrThreadsN == 1 * nrThreadsNrThreadsN ? 1
                    : nrThreadsN % (1 * nrThreadsNrThreadsN) == 0 ? nrThreadsN / (1 * nrThreadsNrThreadsN)
                            : nrThreadsN / (1 * nrThreadsNrThreadsN) + 1;
            kl.launch(nrThreadsNrThreadsN * nrBlocksN, nrWarpsNrThreadsN * 1, 1 * 1, nrThreadsNrThreadsN, nrWarpsNrThreadsN, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrVectorsN = Math.min(16, n);
            int nrThreadsN = n == 1 * nrVectorsN ? 1
                    : n % (1 * nrVectorsN) == 0 ? n / (1 * nrVectorsN) : n / (1 * nrVectorsN) + 1;
            kl.launch(nrVectorsN * nrThreadsN, 1 * 1, 1 * 1, nrVectorsN, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, float[] peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Buffer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, Pointer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, float[] peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, float[] peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Buffer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, Pointer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Buffer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, float[] peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, float[] peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Buffer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Buffer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, int[] indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, int[] indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, int[] indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Buffer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Buffer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Buffer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Pointer indicesPeak,
            float[] input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, float[] input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Pointer indicesPeak,
            Buffer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, Pointer peaks, Pointer indicesPeak,
            Pointer input) throws CashmereNotAvailable {
        launchFindPeakKernel(kl, nrBlocks, n, peak, true, peaks, true, indicesPeak, true, input, true);
    }

    static void launchFindPeakKernel(KernelLaunch kl, int nrBlocks, int n, Pointer peak, boolean copypeak, Pointer peaks,
            boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak, Pointer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.OUT);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int stepSize = nrBlocks * nrThreads;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int nrEls = n / nrThreads;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, float[] peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Buffer peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Pointer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, int[] indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, float[] peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Buffer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Pointer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Buffer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, float[] peaks,
            int[] indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, float[] peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Buffer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Buffer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Pointer peaks,
            int[] indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Pointer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, Pointer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, float[] peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, float[] peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, float[] peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Buffer peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Pointer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, int[] indexPeak, Pointer peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, float[] peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Buffer peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Pointer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Buffer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, float[] peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Buffer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Pointer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, Pointer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Buffer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, float[] peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Buffer peaks, Pointer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Pointer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, int[] indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, int[] indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, float[] peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, float[] peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Buffer peaks, Buffer indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Buffer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Pointer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Pointer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Buffer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Buffer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, float[] peaks,
            int[] indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, float[] peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, float[] peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, float[] peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Buffer peaks, int[] indicesPeak)
            throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Buffer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Buffer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Buffer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Pointer peaks,
            int[] indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, int[] indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Pointer peaks,
            Buffer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Buffer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, Pointer indexPeak, Pointer peaks,
            Pointer indicesPeak) throws CashmereNotAvailable {
        launchMaxLocFloatsKernel(kl, n, peak, true, indexPeak, true, peaks, true, indicesPeak, true);
    }

    static void launchMaxLocFloatsKernel(KernelLaunch kl, int n, Pointer peak, boolean copypeak, Pointer indexPeak,
            boolean copyindexPeak, Pointer peaks, boolean copypeaks, Pointer indicesPeak, boolean copyindicesPeak)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copypeak) {
            kl.setArgument(peak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(peak, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.OUT);
        }
        if (copypeaks) {
            kl.setArgument(peaks, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(peaks, Argument.Direction.IN);
        }
        if (copyindicesPeak) {
            kl.setArgument(indicesPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indicesPeak, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int nrBlocks = 1;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, int[] indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, int[] indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, int[] indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Buffer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Buffer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Buffer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Pointer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Pointer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, Pointer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, double[] energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, int[] indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, int[] indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, int[] indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Buffer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Buffer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Buffer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Pointer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Pointer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, Pointer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Buffer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, int[] indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, int[] indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, int[] indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            int[] indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Buffer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Buffer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Buffer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Buffer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Pointer indexPeak,
            float[] input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, float[] input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Pointer indexPeak,
            Buffer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Buffer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, Pointer indexPeak,
            Pointer input) throws CashmereNotAvailable {
        launchComputeEnergyKernel(kl, nrBlocks, h, w, energy, true, indexPeak, true, input, true);
    }

    static void launchComputeEnergyKernel(KernelLaunch kl, int nrBlocks, int h, int w, Pointer energy, boolean copyenergy,
            Pointer indexPeak, boolean copyindexPeak, Pointer input, boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(nrBlocks, Argument.Direction.IN);
        kl.setArgument(h, Argument.Direction.IN);
        kl.setArgument(w, Argument.Direction.IN);
        if (copyenergy) {
            kl.setArgument(energy, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(energy, Argument.Direction.OUT);
        }
        if (copyindexPeak) {
            kl.setArgument(indexPeak, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(indexPeak, Argument.Direction.IN);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("fermi")) {
            int nrThreads = 256;
            int n = h * w;
            int stepSize = nrThreads * nrBlocks;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = nrBlocks;
            int nrVectors = 16;
            int n = h * w;
            int nrEls = n / nrThreads + 1;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, double[] input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, boolean copyresult, double[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, Buffer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, boolean copyresult, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, Pointer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, double[] result, boolean copyresult, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, double[] input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, boolean copyresult, double[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, Buffer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, boolean copyresult, Buffer input, boolean copyinput)
            throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, Pointer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Buffer result, boolean copyresult, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, double[] input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, boolean copyresult, double[] input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, Buffer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, boolean copyresult, Buffer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, Pointer input) throws CashmereNotAvailable {
        launchSumDoublesKernel(kl, n, result, true, input, true);
    }

    static void launchSumDoublesKernel(KernelLaunch kl, int n, Pointer result, boolean copyresult, Pointer input,
            boolean copyinput) throws CashmereNotAvailable {
        kl.setArgument(n, Argument.Direction.IN);
        if (copyresult) {
            kl.setArgument(result, Argument.Direction.OUT);
        } else {
            kl.setArgumentNoCopy(result, Argument.Direction.OUT);
        }
        if (copyinput) {
            kl.setArgument(input, Argument.Direction.IN);
        } else {
            kl.setArgumentNoCopy(input, Argument.Direction.IN);
        }

        if (kl.getDeviceName().equals("xeon_phi")) {
            int nrThreads = 1;
            int nrVectors = 16;
            kl.launch(nrVectors * nrThreads, 1 * 1, 1 * 1, nrVectors, 1, 1);
        } else if (kl.getDeviceName().equals("fermi")) {
            int nrBlocks = 1;
            int nrThreads = 256;
            int nrThreadsNrThreads = Math.min(32, nrThreads);
            int nrWarpsNrThreads = nrThreads == 1 * nrThreadsNrThreads ? 1
                    : nrThreads % (1 * nrThreadsNrThreads) == 0 ? nrThreads / (1 * nrThreadsNrThreads)
                            : nrThreads / (1 * nrThreadsNrThreads) + 1;
            kl.launch(nrThreadsNrThreads * nrBlocks, nrWarpsNrThreads * 1, 1 * 1, nrThreadsNrThreads, nrWarpsNrThreads, 1);
        } else {
            throw new CashmereNotAvailable("no compatible device found");
        }
    }

}
