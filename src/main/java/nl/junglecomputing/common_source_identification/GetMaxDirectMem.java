/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nl.junglecomputing.common_source_identification;

import java.lang.String;
import java.lang.Long;
import java.util.List;
import java.lang.management.RuntimeMXBean;
import java.lang.management.ManagementFactory;

public class GetMaxDirectMem {
    public static long maxDirectMemory() {
        long retval = -1;
        RuntimeMXBean runtimeMXBean = ManagementFactory.getRuntimeMXBean();
        List<String> vmargs = runtimeMXBean.getInputArguments();
        for (String arg : vmargs) {
            if (arg.startsWith("-XX:MaxDirectMem")) {
                String[] splits = arg.split("=");
                if (splits.length > 1) {
                    char s = splits[1].charAt(splits[1].length() - 1);
                    long factor = 1;
                    int index = "KkMmGg".indexOf(s);
                    String v = splits[1];
                    if (index != -1) {
                        if (index >= 4) {
                            factor = 1024 * 1024 * 1024;
                        } else if (index >= 2) {
                            factor = 1024 * 1024;
                        } else {
                            factor = 1024;
                        }
                        v = v.substring(0, v.length()-1);
                    }
                    try {
                        retval = factor * Long.parseLong(v);
                    } catch(Throwable e) {
                        // Just give up
                        retval = -1;
                    }
                }
                break;
            }
        }
        if (retval <= 0) {
            retval = Runtime.getRuntime().maxMemory();
        }
        return retval;
    }
}
