package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;

/**
 * How a kernel dispatch traces the world octree: with the octree march, or with the block
 * acceleration structure (hardware ray tracing) and its intersection function table. Holds its own
 * reference to the structure, so a rebuild during a render does not free it.
 */
final class BlockTracing implements AutoCloseable {
    private final long accel;
    private long table = 0;

    BlockTracing(MetalSceneLoader loader) {
        accel = Metal.retain(loader.blockAcceleration());
    }

    /** Kernel defines for this kind of tracing. */
    String[] defines(String[] base) {
        return MetalContext.withAcceleration(base, accel);
    }

    /** Creates the function table for a pipeline compiled with {@link #defines}. */
    void bind(long context, long pipeline, MetalSceneLoader loader) {
        if (accel == 0) return;
        Metal.release(table);
        // HwRt_firstBlock only reads the box data of the acceleration structure.
        table = Metal.newFunctionTable(context, pipeline, accel, new long[0], 0);
    }

    /** The kernel's buffers with the acceleration structure and function table inserted after the world octree. */
    long[] buffers(long[] beforeAccel, long[] afterAccel) {
        int extra = accel == 0 ? 0 : 2;
        long[] result = new long[beforeAccel.length + extra + afterAccel.length];
        System.arraycopy(beforeAccel, 0, result, 0, beforeAccel.length);
        if (extra != 0) {
            result[beforeAccel.length] = accel;
            result[beforeAccel.length + 1] = table;
        }
        System.arraycopy(afterAccel, 0, result, beforeAccel.length + extra, afterAccel.length);
        return result;
    }

    @Override
    public void close() {
        Metal.release(table);
        Metal.release(accel);
    }
}
