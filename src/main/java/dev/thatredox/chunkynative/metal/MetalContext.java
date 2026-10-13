package dev.thatredox.chunkynative.metal;

import dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;
import se.llbit.log.Log;

/**
 * The Metal device and the render kernel, compiled from the same source as the OpenCL kernel
 * (see kernel/include/compat.h). Created on first use.
 */
public class MetalContext {
    public final long context;
    public final long renderPipeline;
    public final MetalSceneLoader sceneLoader;

    private static volatile MetalContext instance;

    private MetalContext() {
        context = Metal.createContext();
        long t0 = System.currentTimeMillis();
        // Safe (IEEE) math to match OpenCL, which the kernel was written and tested against.
        renderPipeline = Metal.compile(context, KernelLoader.loadSource("kernel", "rayTracer.c"), "render", false);
        Log.infof("ChunkyCL Metal: %s, render kernel compiled in %d ms",
                Metal.deviceName(context), System.currentTimeMillis() - t0);
        sceneLoader = new MetalSceneLoader(context);
    }

    public static MetalContext get() {
        if (instance == null) {
            synchronized (MetalContext.class) {
                if (instance == null) {
                    instance = new MetalContext();
                }
            }
        }
        return instance;
    }

    /** Whether the Metal backend can run here: macOS with the bridge library bundled. */
    public static boolean isSupported() {
        if (!System.getProperty("os.name").toLowerCase().contains("mac")) {
            return false;
        }
        try {
            Class.forName(Metal.class.getName());
            return true;
        } catch (Throwable e) {
            Log.info("ChunkyCL Metal backend unavailable", e);
            return false;
        }
    }
}
