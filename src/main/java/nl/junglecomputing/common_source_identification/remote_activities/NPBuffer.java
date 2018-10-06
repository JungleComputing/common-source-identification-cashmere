package nl.junglecomputing.common_source_identification.remote_activities;

import java.nio.ByteBuffer;
import java.util.List;

import ibis.cashmere.constellation.Buffer;
import ibis.constellation.util.ByteBuffers;

/**
 * Object to transfer an array of {@link Buffer}s. We need this because neither {@link java.nio.ByteBuffer} nor {@link Buffer} are
 * serializable.
 */
class NPBuffer implements ByteBuffers, java.io.Serializable {

    private static final long serialVersionUID = 1L;

    transient Buffer[] buf;
    int[] indices;

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
