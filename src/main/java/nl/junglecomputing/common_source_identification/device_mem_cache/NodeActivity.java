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

package nl.junglecomputing.common_source_identification.device_mem_cache;

import java.io.File;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ibis.constellation.Activity;
import ibis.constellation.ActivityIdentifier;
import ibis.constellation.Context;

abstract class NodeActivity extends Activity {

    private static final long serialVersionUID = 1L;

    static Logger logger = LoggerFactory.getLogger("CommonSourceIdentification.NodeActivity");

    static final String LABEL = "NodeActivity";

    protected ActivityIdentifier parent;

    protected List<String> nodes;
    protected File[] imageFiles;
    protected int nrImages;
    protected int h;
    protected int w;
    protected boolean mc;

    protected int nodeIndex;
    protected int startIndex;
    protected int endIndex;

    NodeActivity(int h, int w, List<String> nodes, int nodeIndex, File[] imageFiles, boolean mc) {
        super(new Context(nodes.get(nodeIndex) + LABEL), true, true);

        this.parent = null;

        this.nodes = nodes;
        this.imageFiles = imageFiles;
        this.nrImages = imageFiles.length;
        this.h = h;
        this.w = w;
        this.mc = mc;

        this.nodeIndex = nodeIndex;

        int nrImages = imageFiles.length;
        int nrImagesPerNode = getNrImagesPerNode(nrImages, nodes.size());

        this.startIndex = nodeIndex * nrImagesPerNode;
        this.endIndex = getEndIndex(startIndex, nrImagesPerNode, nrImages);
    }

    NodeActivity setParent(ActivityIdentifier aid) {
        this.parent = aid;
        return this;
    }

    static int getNrImagesPerNode(int nrImages, int nrNodes) {
        return (int) Math.ceil((double) nrImages / nrNodes);
    }

    static int getStartIndex(int nodeIndex, int nrImages, int nrNodes) {
        return nodeIndex * getNrImagesPerNode(nrImages, nrNodes);
    }

    static int getEndIndex(int startIndex, int nrImagesInRange, int nrImages) {
        return Math.min(startIndex + nrImagesInRange, nrImages);
    }
}
