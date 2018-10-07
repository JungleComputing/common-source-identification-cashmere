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

import java.time.Duration;
import java.util.Timer;
import java.util.TimerTask;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Activity;
import ibis.constellation.Constellation;
import ibis.constellation.Context;
import ibis.constellation.Event;

/*
 * This activity does not do actual work but keeps track of the progress of the correlations and estimates the time it takes to
 * complete all correlations.  It does this by keeping track of the time of startup, finding out how many correlations have been
 * done compared to all needed correlations.  The time this took is an estimate of how long it will take.  We also compare the
 * throughput of number processed correlations against those of the previous period.  The activity will periodically report the
 * progress.
 */

public class ProgressActivity extends Activity {

    private static final long serialVersionUID = 1L;

    public static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.ProgressActivity");

    public static final String LABEL = "ProgressActivity";
    static final int PERIOD = 10000; // 10 seconds

    // administration of the number of correlations
    private int nrCorrelationsToReceive;
    private int nrReceivedCorrelations;

    // administration for computing the throughput in the last period
    private int nrReceivedCorrelationsPreviously;
    private long timePreviously;

    // administration of the start time
    private Timer timer;
    private long startMillis;

    private class ProgressTask extends TimerTask {
        @Override
        public void run() {
            synchronized (timer) {
                double ratio = nrReceivedCorrelations / (double) nrCorrelationsToReceive;
                String message = "Percentage of correlations: " + Math.round(ratio * 100) + "%";
                if (nrReceivedCorrelations > 0) {
                    long now = System.currentTimeMillis();
                    long elapsedMillis = now - startMillis;
                    long estimatedMillis = (long) (elapsedMillis / ratio) - elapsedMillis;
                    Duration d = Duration.ofMillis(estimatedMillis);
                    message += ", estimated time: " + format(d);
                    if (nrReceivedCorrelationsPreviously > 0) {
                        int nrCorrelationsThisPeriod = nrReceivedCorrelations - nrReceivedCorrelationsPreviously;
                        double period = (now - timePreviously) / 1000.0;
                        double throughput = nrCorrelationsThisPeriod / period;
                        message += String.format(", througput: %.2f correlations/s", throughput);
                    }
                    nrReceivedCorrelationsPreviously = nrReceivedCorrelations;
                    timePreviously = now;
                } else {
                    startMillis = System.currentTimeMillis();
                }
                logger.info(message);
                if (logger.isDebugEnabled()) {
                    logger.debug("Received {}/{} notifications", nrReceivedCorrelations, nrCorrelationsToReceive);
                }
            }
        }
    }

    public static String format(Duration duration) {
        return duration.toString().substring(2).replaceAll("(\\d[HMS])(?!$)", "$1 ").toLowerCase();
    }

    public ProgressActivity(int nrCorrelationsToReceive) {
        super(new Context(ProgressActivity.LABEL), true, true);

        this.nrCorrelationsToReceive = nrCorrelationsToReceive;
        this.nrReceivedCorrelations = 0;
        this.nrReceivedCorrelationsPreviously = 0;

        this.timer = new Timer();
        this.startMillis = System.currentTimeMillis();
    }

    @Override
    public int initialize(Constellation cons) {
        if (logger.isDebugEnabled()) {
            logger.debug("Starting the progress task");
        }

        timer.schedule(new ProgressTask(), 0, PERIOD);
        startMillis = System.currentTimeMillis();
        return SUSPEND;
    }

    @Override
    public int process(Constellation cons, Event event) {
        int nrCorrelations = (Integer) event.getData();
        synchronized (timer) {
            if (logger.isDebugEnabled()) {
                logger.debug("Received notification of {} correlations", nrCorrelations);
            }
            nrReceivedCorrelations += nrCorrelations;
        }
        if (nrReceivedCorrelations == nrCorrelationsToReceive) {
            return FINISH;
        } else {
            return SUSPEND;
        }
    }

    @Override
    public void cleanup(Constellation cons) {
        timer.cancel();
        System.out.println();
    }
}
