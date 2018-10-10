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

package nl.junglecomputing.common_source_identification.multipleGPUs;

import java.io.File;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;

/**
 * This activity's only task is to send a request to a GetNoisePatternsActivity, wait for its answer, and send that to the
 * TreeCorrelationActivity needing this data. We have a separate activity for that, so that the TreeCorrelationActivity can do a
 * wait(), instead of a suspend, which would cause it to start a new correlation, which we don't want because there are not enough
 * resources to do so.
 */

public class FetchPatternActivity extends Activity {

    private static final long serialVersionUID = 1L;
    static final String LABEL = "FetchPattern";

    private transient final CorrelationsActivity lca;
    final GetNoisePatternsActivity.PatternsInfo request = new GetNoisePatternsActivity.PatternsInfo();
    private final ActivityIdentifier target;

    public FetchPatternActivity(File[] files, int[] indices, CorrelationsActivity lca, ActivityIdentifier target) {
        super(new Context(LABEL, 1), true);
        request.indices = indices;
        request.files = files;
        this.target = target;
        this.lca = lca;
    }

    @Override
    public int initialize(Constellation constellation) {
        request.target = identifier(); // can only be initialized here, since in the constructor, the activity identifier does not exist yet.
        constellation.send(new Event(identifier(), target, request));
        return SUSPEND;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        lca.pushMessage(event, this);
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // nothing
    }
}
