package dev.thatredox.chunkynative.metal.util;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.util.NativeCleaner;
import it.unimi.dsi.fastutil.ints.IntArrayList;

/**
 * A Metal buffer in shared memory, written and read by copying from and to Java arrays at an index
 * (so it can be larger than 2 GB). Only access it while no kernel using it is running. Released on
 * {@link #close()}, or when garbage collected.
 */
public class MetalBuffer implements AutoCloseable {
    public final long handle;
    private final NativeCleaner.Cleaner cleaner;

    public MetalBuffer(long context, long bytes) {
        long max = Metal.maxBufferLength(context);
        if (bytes > max) {
            throw new IllegalArgumentException(String.format(
                    "The scene needs a %.2f GB buffer, but this GPU allows at most %.2f GB. Load fewer chunks.",
                    bytes / 1e9, max / 1e9));
        }
        long handle = Metal.newBuffer(context, Math.max(bytes, 4));
        this.handle = handle;
        this.cleaner = NativeCleaner.INSTANCE.register(this, () -> Metal.release(handle));
    }

    /** A buffer holding the first {@code length} values of {@code data} (at least one int, like ClIntBuffer). */
    public static MetalBuffer ofInts(long context, int[] data, int length) {
        MetalBuffer buffer = new MetalBuffer(context, 4L * length);
        buffer.writeInts(0, data, 0, length);
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
        buffer.writeFloats(0, data, 0, data.length);
        return buffer;
    }

    /** Writes {@code length} ints from {@code data[offset]} to the buffer starting at int {@code index}. */
    public void writeInts(long index, int[] data, int offset, int length) {
        Metal.writeInts(handle, 4 * index, data, offset, length);
    }

    public void writeInt(long index, int value) {
        writeInts(index, new int[] {value}, 0, 1);
    }

    /** Writes {@code length} floats from {@code data[offset]} to the buffer starting at float {@code index}. */
    public void writeFloats(long index, float[] data, int offset, int length) {
        Metal.writeFloats(handle, 4 * index, data, offset, length);
    }

    /** Reads {@code data.length} ints from the start of the buffer. */
    public void readInts(int[] data) {
        Metal.readInts(handle, 0, data, 0, data.length);
    }

    /** Reads {@code data.length} floats from the start of the buffer. */
    public void readFloats(float[] data) {
        Metal.readFloats(handle, 0, data, 0, data.length);
    }

    @Override
    public void close() {
        cleaner.clean();
    }
}
