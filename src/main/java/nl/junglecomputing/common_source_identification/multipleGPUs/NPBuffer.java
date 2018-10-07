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

import java.nio.ByteBuffer;
import java.util.List;

import ibis.cashmere.constellation.Buffer;
import ibis.constellation.util.ByteBuffers;

/**
 * Object to transfer an array of {@link Buffer}s. We need this because neither {@link java.nio.ByteBuffer} nor {@link Buffer} are
 * serializable.
 */
public class NPBuffer implements ByteBuffers, java.io.Serializable {

    private static final long serialVersionUID = 1L;

    public transient Buffer[] buf;
    public int[] indices;

    public NPBuffer(Buffer[] availableElements, int[] indices) {
        buf = availableElements;
        this.indices = indices;
    }

    @Override
    public void pushByteBuffers(List<ByteBuffer> list) {
        for (Buffer b : buf) {
            ByteBuffer bb = b.getByteBuffer();
            bb.rewind(); // Just to be sure
            list.add(bb);
        }
    }

    @Override
    public void popByteBuffers(List<ByteBuffer> list) {
        buf = new Buffer[list.size()];
        for (int i = 0; i < buf.length; i++) {
            ByteBuffer b = list.remove(0);
            b.rewind();
            buf[i] = new Buffer(b);
        }
    }

}
