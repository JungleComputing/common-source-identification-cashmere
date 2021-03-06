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

import java.nio.ByteBuffer;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.constellation.Timer;

class ImageToGrayscaleStage extends Stage {

    static float[] execute(Buffer image, int h, int w, String executor) {
        Timer timer = Cashmere.getTimer("java", executor, "convert to grayscale");

        int event = timer.start();
        ByteBuffer buf = image.getByteBuffer();

        float[] pixelsFloat = new float[h * w];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float r = buf.get((i * w + j) * 3 + 0) & 0xff;
                float g = buf.get((i * w + j) * 3 + 1) & 0xff;
                float b = buf.get((i * w + j) * 3 + 2) & 0xff;
                pixelsFloat[i * w + j] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }

        timer.stop(event);

        return pixelsFloat;
    }
}
