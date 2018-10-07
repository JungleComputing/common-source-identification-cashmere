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

package nl.junglecomputing.common_source_identification.remote_activities;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import nl.junglecomputing.common_source_identification.mc.ComputeFrequencyDomain;
import nl.junglecomputing.common_source_identification.mc.ComputeNoisePattern;

class GetStatsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    private final ActivityIdentifier parent;

    public GetStatsActivity(ActivityIdentifier id, int host) {
        super(new Context(GetNoisePatternsActivity.LABEL + host, 1), false);
        this.parent = id;
    }

    @Override
    public int initialize(Constellation constellation) {
        constellation.send(new Event(identifier(), parent, new int[] { ComputeNoisePattern.patternsComputed.getAndSet(0),
                ComputeFrequencyDomain.transformed.getAndSet(0), GetNoisePatternsActivity.countFetched.getAndSet(0) }));
        return FINISH;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // nothing
    }
}
