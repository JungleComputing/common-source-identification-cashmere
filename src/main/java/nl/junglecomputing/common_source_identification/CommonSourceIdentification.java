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

import ibis.constellation.NoSuitableExecutorException;

public class CommonSourceIdentification {

    public static final String USAGE = "Usage: java CommonSourceIdentification -image-dir <image-dir> [ -cpu -mc -mainMemCache -deviceMemCache -remote-activities -dedicated-activities -relaxed -multipleGPUs ]";

    public static void main(String[] args) throws NoSuitableExecutorException {
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-cpu")) {
                nl.junglecomputing.common_source_identification.cpu.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-mc")) {
                nl.junglecomputing.common_source_identification.mc.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-mainMemCache")) {
                nl.junglecomputing.common_source_identification.main_mem_cache.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-deviceMemCache")) {
                nl.junglecomputing.common_source_identification.device_mem_cache.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-remote-activities")) {
                nl.junglecomputing.common_source_identification.remote_activities.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-dedicated-activities")) {
                nl.junglecomputing.common_source_identification.dedicated_activities.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-relaxed")) {
                nl.junglecomputing.common_source_identification.relaxed.CommonSourceIdentification.main(args);
                break;
            } else if (args[i].equals("-multipleGPUs")) {
                nl.junglecomputing.common_source_identification.multipleGPUs.CommonSourceIdentification.main(args);
                break;
            } else {
                throw new Error(USAGE);
            }
        }
    }
}
