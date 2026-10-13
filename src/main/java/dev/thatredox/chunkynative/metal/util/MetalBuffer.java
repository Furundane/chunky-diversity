package dev.thatredox.chunkynative.metal.util;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.util.NativeCleaner;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import java.nio.ByteBuffer;

/**
 * A Metal buffer in shared memory. Write and read it through {@link #contents()} while no kernel
 * using it is running. Released on {@link #close()}, or when garbage collected.
 */
public class MetalBuffer implements AutoCloseable {
    public final long handle;
    private final ByteBuffer contents;
    private final NativeCleaner.Cleaner cleaner;

    public MetalBuffer(long context, long bytes) {
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Metal buffers larger than 2 GB are not supported: " + bytes);
        }
        long handle = Metal.newBuffer(context, Math.max(bytes, 4));
        this.handle = handle;
        this.contents = Metal.contents(handle);
        this.cleaner = NativeCleaner.INSTANCE.register(this, () -> Metal.release(handle));
    }

    /** A buffer holding the first {@code length} values of {@code data} (at least one int, like ClIntBuffer). */
    public static MetalBuffer ofInts(long context, int[] data, int length) {
        MetalBuffer buffer = new MetalBuffer(context, 4L * length);
        buffer.contents.asIntBuffer().put(data, 0, length);
        return buffer;
    }

    public static MetalBuffer ofInts(long context, int[] data) {
        return ofInts(context, data, data.length);
    }

    public static MetalBuffer ofInts(long context, IntArrayList data) {
        return ofInts(context, data.elements(), data.size());
    }

    public static MetalBuffer ofInt(long context, int value) {
        return ofInts(context, new int[] {value});
    }

    public static MetalBuffer ofFloats(long context, float[] data) {
        MetalBuffer buffer = new MetalBuffer(context, 4L * data.length);
        buffer.contents.asFloatBuffer().put(data);
        return buffer;
    }

    /** Native-order view of the buffer's memory. */
    public ByteBuffer contents() {
        return contents;
    }

    @Override
    public void close() {
        cleaner.clean();
    }
}
