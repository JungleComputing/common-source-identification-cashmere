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

import org.jocl.Pointer;

import ibis.cashmere.constellation.Buffer;
import ibis.cashmere.constellation.Cashmere;
import ibis.cashmere.constellation.CashmereNotAvailable;
import ibis.cashmere.constellation.Device;
import ibis.cashmere.constellation.LibFuncNotAvailable;
import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;

public class CorrelationsActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static boolean FLIPPED = true;
    static boolean REGULAR = false;

    public static final String LABEL = "Correlation";
    private ActivityIdentifier id;
    private int height;
    private int width;
    private int i;
    private int j;
    private File fi;
    private File fj;

    public CorrelationsActivity(ActivityIdentifier id, int height, int width, int i, int j, File fi, File fj) {
        super(new Context(LABEL, 1), false);
        this.id = id;
        this.height = height;
        this.width = width;
        this.i = i;
        this.j = j;
        this.fi = fi;
        this.fj = fj;
    }

    @Override
    public int initialize(Constellation constellation) {
        Correlation c = null;
        String executor = constellation.identifier().toString();
	try {
	    c = ComputeCPU.computeCorrelation(height, width, i, j, fi, fj, executor);
	} catch (IOException e) {
	    throw new Error(e);
	}
        constellation.send(new Event(identifier(), id, c));
        return FINISH;
    }

    @Override
    public int process(Constellation constellation, Event event) {
        return FINISH;
    }

    @Override
    public void cleanup(Constellation constellation) {
        // empty
    }
}
