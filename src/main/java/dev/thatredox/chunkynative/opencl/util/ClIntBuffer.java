package dev.thatredox.chunkynative.opencl.util;

import dev.thatredox.chunkynative.common.export.Packer;
import dev.thatredox.chunkynative.opencl.context.ClContext;

import static org.jocl.CL.*;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.jocl.*;

public class ClIntBuffer implements AutoCloseable {
    private final ClMemory buffer;
    private final ClContext context;

    public ClIntBuffer(int[] buffer, int length, ClContext context) {
        if (length == 0) {
            buffer = new int[1];
            length = 1;
        }
        assert buffer.length >= length;

        this.context = context;
        this.buffer = new ClMemory(
                clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_uint * length, Pointer.to(buffer), null));
    }

    /** An uninitialized buffer of {@code length} ints (at least one), to fill with {@link #set(int[], int, long)}. */
    public ClIntBuffer(long length, ClContext context) {
        this.context = context;
        this.buffer = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY,
                (long) Sizeof.cl_uint * Math.max(length, 1), null, null));
    }

    public ClIntBuffer(int[] buffer, ClContext context) {
        this(buffer, buffer.length, context);
    }

    public ClIntBuffer(IntArrayList buffer, ClContext context) {
        this(buffer.elements(), buffer.size(), context);
    }

    public ClIntBuffer(Packer packable, ClContext context) {
        this(packable.pack(), context);
    }

    public ClIntBuffer(int value, ClContext context) {
        this(new int[] {value}, context);
    }

    public cl_mem get() {
        return buffer.get();
    }

    public void set(int[] values, int offset) {
        clEnqueueWriteBuffer(context.queue, this.get(), CL_TRUE, (long) Sizeof.cl_uint * offset,
                (long) Sizeof.cl_uint * values.length, Pointer.to(values),
                0, null, null);
    }

    /** Writes the first {@code count} values to the buffer starting at int {@code offset}. */
    public void set(int[] values, int count, long offset) {
        clEnqueueWriteBuffer(context.queue, this.get(), CL_TRUE, (long) Sizeof.cl_uint * offset,
                (long) Sizeof.cl_uint * count, Pointer.to(values),
                0, null, null);
    }

    @Override
    public void close() {
        buffer.close();
    }
}
