package nl.junglecomputing.common_source_identification.cpu;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.List;

import java.lang.management.ManagementFactory;

import ibis.cashmere.constellation.Cashmere;
import ibis.constellation.NoSuitableExecutorException;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.util.SingleEventCollector;


public class NodeInformation {

    // constants for setting up Constellation (some are package private)
    static String HOSTNAME = "localhost";
    static int ID = 0;
    static String STEALPOOL = "stealpool";
    static String LABEL = "commonSourceIdentification";


    /*
     * All kinds of bookkeeping methods
     */

    static void setNodeID(List<String> nodes) {
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).equals(HOSTNAME)) {
                ID = i;
                return;
            }
        }
    }

    static int getNrExecutors(String property, int defaultValue) {
        String prop = System.getProperties().getProperty(property);
        int nrExecutors = defaultValue;
        if (prop != null) {
            nrExecutors = Integer.parseInt(prop);
        }

        return nrExecutors;
    }

    static void setHostName() {
        try {
            Runtime r = Runtime.getRuntime();
            Process p = r.exec("hostname");
            p.waitFor();
            BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
            HOSTNAME = b.readLine();
            b.close();
        } catch (IOException | InterruptedException e) {
        }
    }

    
    /*
     * Various Constellation activities
     */

    static ActivityIdentifier progressActivity(SingleEventCollector sec, int nrImages) throws NoSuitableExecutorException {

        ActivityIdentifier aid = Cashmere.submit(sec);

        ProgressActivity progressActivity = new ProgressActivity(aid, nrImages);

        ActivityIdentifier progressActivityID = Cashmere.submit(progressActivity);

        return progressActivityID;
    }

    static String getProcessId(final String fallback) {
        // Note: may fail in some JVM implementations
        // therefore fallback has to be provided

        // something like '<pid>@<hostname>', at least in SUN / Oracle JVMs
        final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
        final int index = jvmName.indexOf('@');

        if (index < 1) {
            // part before '@' empty (index = 0) / '@' not found (index = -1)
            return fallback;
        }

        try {
            return Long.toString(Long.parseLong(jvmName.substring(0, index)));
        } catch (NumberFormatException e) {
            // ignore
        }
        return fallback;
    }

}
