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

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.constellation.Timer;

class FileToImageStage extends Stage {

    static BufferedImage readImage(File file) throws IOException {

        final InputStream fileInputStream = new FileInputStream(file);
        BufferedImage image = null;
        synchronized (FileToImageStage.class) {
            image = ImageIO.read(new BufferedInputStream(fileInputStream));
        }
        if ((image != null) && (image.getWidth() >= 0) && (image.getHeight() >= 0)) {
            return image;
        }
        return null;
    }

    static BufferedImage execute(File file, String executor) throws IOException {

        Timer timer = Cashmere.getTimer("java", executor, "Read in image");
        int event = timer.start();
        BufferedImage image = readImage(file);
        timer.stop(event);
        return image;
    }

    static Buffer execute(String filename, int h, int w, String executor, Buffer data) throws IOException {

        Timer timer = Cashmere.getTimer("java", executor, "Read in image");
        int event = timer.start();

        ReadJPG.readJPG(data, filename);

        timer.stop(event);

        return data;
    }
}
