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

import java.io.File;

import ibis.constellation.Activity;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;
import ibis.constellation.NoSuitableExecutorException;

public class FetchPatternActivity extends Activity {

    private static final long serialVersionUID = 1L;
    static final String LABEL = "FetchPattern";

    private final int[] indices;
    private final int loc_indices;
    private transient final LeafCorrelationsActivity lca;
    private final File[] files;
    private final int height;
    private final int width;

    public FetchPatternActivity(File[] files, int height, int width, int[] indices, int loc_indices,
            LeafCorrelationsActivity lca) {
        super(new Context(LABEL, 1), true);
        this.indices = indices;
        this.loc_indices = loc_indices;
        this.lca = lca;
        this.files = files;
        this.height = height;
        this.width = width;
    }

    @Override
    public int initialize(Constellation constellation) {
        try {
            constellation.submit(new GetNoisePatternsActivity(identifier(), files, height, width, indices, loc_indices));
        } catch (NoSuitableExecutorException e) {
            throw new Error("Should not happen", e);
        }
        return SUSPEND;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        lca.pushMessage(event);
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // nothing
    }
}
