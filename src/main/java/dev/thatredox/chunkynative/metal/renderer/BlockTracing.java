package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;

import java.util.HashMap;
import java.util.Map;

/**
 * How a kernel dispatch traces the world octree: with the octree march, or with the block
 * acceleration structure (hardware ray tracing) and its intersection function table. Holds its own
 * reference to the structure, so a rebuild during a render does not free it.
 */
final class BlockTracing implements AutoCloseable {
    private final long accel;
    /** Intersection function table per pipeline. */
    private final Map<Long, Long> tables = new HashMap<>();

    BlockTracing(MetalSceneLoader loader) {
        accel = Metal.retain(loader.blockAcceleration());
    }

    /** Kernel defines for this kind of tracing. */
    String[] defines(String[] base) {
        return MetalContext.withAcceleration(base, accel);
    }

    /** Creates the function table for a pipeline compiled with {@link #defines}. */
    void bind(long context, long pipeline) {
        if (accel == 0 || tables.containsKey(pipeline)) return;
        // HwRt_firstBlock only reads the box data of the acceleration structure.
        tables.put(pipeline, Metal.newFunctionTable(context, pipeline, accel, new long[0], 0));
    }

    /**
     * The buffers of a kernel run with a pipeline (bound with {@link #bind}): the acceleration
     * structure and function table are inserted after the world octree.
     */
    long[] buffers(long pipeline, long[] beforeAccel, long[] afterAccel) {
        int extra = accel == 0 ? 0 : 2;
        long[] result = new long[beforeAccel.length + extra + afterAccel.length];
        System.arraycopy(beforeAccel, 0, result, 0, beforeAccel.length);
        if (extra != 0) {
            result[beforeAccel.length] = accel;
            result[beforeAccel.length + 1] = tables.get(pipeline);
        }
        System.arraycopy(afterAccel, 0, result, beforeAccel.length + extra, afterAccel.length);
        return result;
    }

    @Override
    public void close() {
        for (long table : tables.values()) Metal.release(table);
        Metal.release(accel);
    }
}
