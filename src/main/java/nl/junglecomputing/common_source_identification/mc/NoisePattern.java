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

package nl.junglecomputing.common_source_identification.mc;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.List;

import ibis.cashmere.constellation.Buffer;
import ibis.constellation.util.ByteBuffers;

class NoisePattern implements Serializable, ByteBuffers {

    private static final long serialVersionUID = 1L;

    int index;
    boolean flipped;
    transient Buffer buffer;

    NoisePattern(int index, boolean flipped, Buffer buffer) {
        this.index = index;
        this.flipped = flipped;
        this.buffer = buffer;
    }

    public void pushByteBuffers(List<ByteBuffer> byteBuffers) {
        if (buffer != null) {
            byteBuffers.add(buffer.getByteBuffer());
        }
    }

    public void popByteBuffers(List<ByteBuffer> byteBuffers) {
        if (byteBuffers.size() > 0) {
            buffer = new Buffer(byteBuffers.remove(0));
        }
    }
}
