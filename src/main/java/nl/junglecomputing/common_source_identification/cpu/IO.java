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
import java.io.FileNotFoundException;
import java.io.PrintStream;

import java.util.Arrays;
import java.util.Comparator;

import nl.junglecomputing.common_source_identification.Version;

public class IO {

    public static void writeFiles(CorrelationMatrix correlationMatrix, File[] imageFiles, Version version) throws FileNotFoundException {
        PrintStream out = new PrintStream("prnu_" + version + ".out");

        double[][] coefficients = correlationMatrix.coefficients;

        for (int i = 0; i < coefficients.length; i++) {
            for (int j = 0; j < coefficients.length; j++) {
                out.printf("%.6f, ", coefficients[i][j]);
            }
            out.println();
        }
        out.close();
    }
    

    public static File[] getImageFiles(String nameImageDir) throws IOException {
        File imageDir = new File(nameImageDir);
        if (!(imageDir.exists() && imageDir.isDirectory())) {
            throw new IOException(nameImageDir + " is not a valid directory");
        }
        File[] imageFiles = imageDir.listFiles();
        Arrays.sort(imageFiles, new Comparator<File>() {
            public int compare(File f1, File f2) {
                return f1.getName().compareTo(f2.getName());
            }
        });
        return imageFiles;
    }


}
