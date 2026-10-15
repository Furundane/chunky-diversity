package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;
import dev.thatredox.chunkynative.metal.MetalSettings;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * How a kernel dispatch traces the world octree: with the octree march, or with the block
 * acceleration structure (hardware ray tracing) and its intersection function table. Holds its own
 * reference to the structure, so a rebuild during a render does not free it.
 */
final class BlockTracing implements AutoCloseable {
    /** Kernel define for hardware ray tracing of the water octree (wavefront kernels only). */
    static final String WATER_DEFINE = "CHUNKYCL_HWRT_WATER";

    private final long accel;
    private final long waterAccel;
    /** Intersection function tables per pipeline. */
    private final Map<Long, Long> tables = new HashMap<>();
    private final Map<Long, Long> waterTables = new HashMap<>();

    BlockTracing(MetalSceneLoader loader) {
        this(loader, false);
    }

    /**
     * water: allow tracing the water octree with hardware ray tracing (the wavefront kernels can),
     * if {@link MetalSettings#hardwareWater} is on. Off by default: on the test worlds it was no
     * faster, since most of the water cost is in rays inside water, which use the octree march.
     */
    BlockTracing(MetalSceneLoader loader, boolean water) {
        accel = Metal.retain(loader.blockAcceleration());
        waterAccel = water && accel != 0 && MetalSettings.hardwareWater()
                ? Metal.retain(loader.waterAcceleration()) : 0;
    }

    /** Whether the water octree is traced with hardware ray tracing. */
    boolean hardwareWater() {
        return waterAccel != 0;
    }

    /** Whether the world octree is traced with hardware ray tracing. */
    boolean hardware() {
        return accel != 0;
    }

    /** Kernel defines for this kind of tracing. */
    String[] defines(String[] base) {
        String[] defines = MetalContext.withAcceleration(base, accel);
        if (waterAccel == 0) return defines;
        String[] result = Arrays.copyOf(defines, defines.length + 1);
        result[defines.length] = WATER_DEFINE;
        return result;
    }

    /** Creates the function table for a pipeline compiled with {@link #defines}. */
    void bind(long context, long pipeline) {
        if (accel == 0 || tables.containsKey(pipeline)) return;
        // HwRt_firstBlock only reads the box data of the acceleration structure.
        tables.put(pipeline, Metal.newFunctionTable(context, pipeline, accel, new long[0], 0));
        if (waterAccel != 0) waterTables.put(pipeline, Metal.newFunctionTable(context, pipeline, waterAccel, new long[0], 0));
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

    /**
     * Like {@link #buffers(long, long[], long[])} for the wavefront kernels, which also take the
     * water acceleration structure and function table after the water octree.
     */
    long[] buffers(long pipeline, long[] beforeAccel, long[] beforeWaterAccel, long[] afterWaterAccel) {
        long[] first = buffers(pipeline, beforeAccel, beforeWaterAccel);
        int extra = waterAccel == 0 ? 0 : 2;
        long[] result = new long[first.length + extra + afterWaterAccel.length];
        System.arraycopy(first, 0, result, 0, first.length);
        if (extra != 0) {
            result[first.length] = waterAccel;
            result[first.length + 1] = waterTables.get(pipeline);
        }
        System.arraycopy(afterWaterAccel, 0, result, first.length + extra, afterWaterAccel.length);
        return result;
    }

    @Override
    public void close() {
        for (long table : tables.values()) Metal.release(table);
        for (long table : waterTables.values()) Metal.release(table);
        Metal.release(accel);
        Metal.release(waterAccel);
    }
}
