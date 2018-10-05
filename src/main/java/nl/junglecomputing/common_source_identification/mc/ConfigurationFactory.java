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

import java.util.ArrayList;

import ibis.constellation.AbstractContext;
import ibis.constellation.ConstellationConfiguration;
import ibis.constellation.Context;
import ibis.constellation.StealPool;
import ibis.constellation.StealStrategy;

/*
 * Helper class to make Constellation Configurations.
 */
class ConfigurationFactory {

    private ArrayList<ConstellationConfiguration> configurations;

    ConfigurationFactory() {
        configurations = new ArrayList<ConstellationConfiguration>();
    }

    void createConfigurations(int nrExecutors, StealPool myPool, StealPool stealsFrom, AbstractContext context,
            StealStrategy localStrategy, StealStrategy remoteStrategy) {

        for (int i = 0; i < nrExecutors; i++) {
            configurations.add(new ConstellationConfiguration(context, myPool, stealsFrom, localStrategy, StealStrategy.SMALLEST,
                    remoteStrategy));
        }
    }

    void createConfigurations(int nrExecutors, StealPool myPool, StealPool stealsFrom, String context,
            StealStrategy localStrategy, StealStrategy remoteStrategy) {
        createConfigurations(nrExecutors, myPool, stealsFrom, new Context(context), localStrategy, remoteStrategy);
    }

    void createConfigurations(int nrExecutors, StealPool myPool, StealPool stealsFrom, String context) {
        createConfigurations(nrExecutors, myPool, stealsFrom, new Context(context), StealStrategy.SMALLEST,
                StealStrategy.SMALLEST);
    }

    ConstellationConfiguration[] getConfigurations() {
        ConstellationConfiguration[] array = new ConstellationConfiguration[configurations.size()];
        return configurations.toArray(array);
    }
}
