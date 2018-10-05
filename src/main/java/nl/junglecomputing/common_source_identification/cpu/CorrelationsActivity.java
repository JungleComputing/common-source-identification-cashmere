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
