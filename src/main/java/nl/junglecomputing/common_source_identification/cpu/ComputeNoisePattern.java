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

package nl.junglecomputing.common_source_identification.cpu;

import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.cashmere.constellation.Buffer;

class ComputeNoisePattern {

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.ComputeNoisePattern");

    static float[] computePRNU(int index, File file, int h, int w, String executor) throws IOException {
        float[] dxs = new float[h * w];
        float[] dys = new float[h * w];
        Buffer bufferHWRGB = new Buffer(h * w * 3);

        FileToImageStage.execute(file.getPath(), h, w, executor, bufferHWRGB);
        float[] grayscale = ImageToGrayscaleStage.execute(bufferHWRGB, h, w, executor);
        float[] withoutNoise = FastNoiseStage.execute(h, w, grayscale, executor, dxs, dys);
        float[] normalized = ZeroMeanStage.execute(h, w, withoutNoise, executor);
        return WienerStage.execute(h, w, normalized, executor);
    }
}
