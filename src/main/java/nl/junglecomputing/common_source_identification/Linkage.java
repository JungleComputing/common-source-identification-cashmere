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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;

import java.io.PrintWriter;

class Linkage {

    static final String LINKAGE_FILENAME = "linkage.txt";
    static final String CLUSTERING_FILENAME = "clustering.txt";
    
    static int[] findMax(double[][] cortable) {
        int N = cortable[0].length;
        double highest = -1e100;
        int index[] = new int[2];
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                if (cortable[i][j] > highest) {
                    highest = cortable[i][j];
                    index[0] = i;
                    index[1] = j;
                }
            }
        }
        return index;
    }

    static ArrayList<Link> hierarchical_clustering(double[][] cortable) {
        int N = cortable.length;
        double[][] matrix = new double[N][N];

        int c = 0;

        //copy cortable into matrix
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                matrix[i][j] = cortable[i][j];
            }
        }

        //create data structures to hold info about clusters
        int next_cluster_id = N;
        ArrayList<Integer> cluster_ids = new ArrayList<Integer>(N);
        for (int i=0; i<N; i++) {
            cluster_ids.add(i,i);
        }

        HashMap<Integer,ArrayList<Integer>> cluster_members = new HashMap<Integer,ArrayList<Integer>>();
        for (int i=0; i<N; i++) {
            ArrayList<Integer> l = new ArrayList<Integer>(1);
            l.add(i);
            cluster_members.put(i, l);
        }

        ArrayList<Link> linkage = new ArrayList<Link>(N-1);

        for (int iterator=0; iterator<N-1; iterator++) {

            //find the most similar pair of clusters
            int[] index_max = findMax(matrix);
            int n1 = index_max[0];
            int n2 = index_max[1];

            if (n1 == n2) {
                break;
            }

            //merge the clusters into a new cluster in our bookkeeping data structures
            int cluster1 = cluster_ids.get(n1);
            int cluster2 = cluster_ids.get(n2);
            ArrayList<Integer> cluster1_members = cluster_members.get(cluster1);
            cluster_members.remove(cluster1);
            ArrayList<Integer> cluster2_members = cluster_members.get(cluster2);
            cluster_members.remove(cluster2);
            cluster1_members.addAll(cluster2_members);
            cluster_members.put(next_cluster_id, cluster1_members);
            cluster_ids.set(n1, next_cluster_id);

            //add to linkage
            int new_size = cluster_members.get(next_cluster_id).size();
            linkage.add(new Link(cluster1, cluster2, matrix[n1][n2], new_size));
            if (new_size >= N) {
                break;
            }

            //update the similarity matrix
            for (int i=0; i<N; i++) {
                if (cluster_members.containsKey(i)) {
                    int other = cluster_ids.get(i);
                    double sum = 0.0;
                    ArrayList<Integer> a = cluster_members.get(next_cluster_id);
                    ArrayList<Integer> b = cluster_members.get(other);

                    for (int j=0; j<a.size(); j++) {
                        for (int k=0; k<b.size(); k++) {
                            sum += cortable[a.get(j)][b.get(k)]; //needs to be cortable NOT matrix
                        }
                    }

                    double avg = sum / (a.size()*b.size());

                    matrix[n1][i] = avg;
                    matrix[i][n1] = avg;
                }
            }

            //erase cluster n2
            for (int i=0; i<N; i++) {
                matrix[n2][i] = -1e200;
                matrix[i][n2] = -1e200;
            }

            //increment next cluster id for next cluster merger
            next_cluster_id += 1;

        }

        return linkage;
    }

    static void write_linkage(ArrayList<Link> linkage) {
        try {
            PrintWriter textfile = new PrintWriter(LINKAGE_FILENAME);
            for (Link l : linkage) {
                textfile.println("[" + l.n1 + "," + l.n2 + "," + l.dist + "," + l.size + "]");
            }
            textfile.println();
            textfile.close();
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    static void write_flat_clustering(ArrayList<Link> linkage, int N) {
        final double THRESHOLD = 60.0;

        try {
            PrintWriter textfile = new PrintWriter(CLUSTERING_FILENAME);
            textfile.println("flat clustering:");

            //create data structures to hold info about clusters
            int next_cluster_id = N;
            HashMap<Integer,ArrayList<Integer>> cluster_members = new HashMap<Integer,ArrayList<Integer>>();
            for (int i=0; i<N; i++) {
                ArrayList<Integer> l = new ArrayList<Integer>(1);
                l.add(i);
                cluster_members.put(i, l);
            }

            boolean termination = false;
            Iterator<Link> link_iterator = linkage.iterator();
            for (int i=0; i<N-1 && termination == false; i++) {
                Link link = link_iterator.next();
                //System.out.println("[" + link.n1 + "," + link.n2 + "," + link.dist + "," + link.size + "]");

                if (link.dist < THRESHOLD) {
                    for (Map.Entry<Integer, ArrayList<Integer>> entry : cluster_members.entrySet()) {
                        ArrayList<Integer> list = entry.getValue();
                        Collections.sort(list);
                        textfile.println(entry.getKey() + "=" + list.toString());
                    }
                    termination = true;
                }
            
                if (termination == false) {
                    //merge the clusters into a new cluster in our bookkeeping data structures
                    int cluster1 = link.n1;
                    int cluster2 = link.n2;
                    ArrayList<Integer> cluster1_members = cluster_members.get(cluster1);
                    cluster_members.remove(cluster1);
                    ArrayList<Integer> cluster2_members = cluster_members.get(cluster2);
                    cluster_members.remove(cluster2);
                    cluster1_members.addAll(cluster2_members);
                    cluster_members.put(next_cluster_id, cluster1_members);
                    next_cluster_id += 1;
                }
            }

            textfile.println();
            textfile.flush();
            textfile.println("labels:");

            int[] labeling = new int[N];
            for (int i=0; i<N; i++) {
                labeling[i] = 0;
            }

            int label = 1;
            for (Map.Entry<Integer, ArrayList<Integer>> entry : cluster_members.entrySet()) {
                //System.out.println("label=" + label + "key=" + entry.getKey() + "value=" + entry.getValue().toString() );
                for (Integer m: entry.getValue()) {
                    labeling[m.intValue()] = label;
                }
                label += 1;
            }

            int num_digits = (int)Math.log10(label)+1;
            String format = "%"+num_digits+"d ";
            textfile.print("[");
            for (int l: labeling) {
                textfile.format(format, l);
            }
            textfile.println("]");

            textfile.println();
            textfile.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}


