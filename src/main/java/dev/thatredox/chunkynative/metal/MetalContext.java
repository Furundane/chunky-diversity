package dev.thatredox.chunkynative.metal;

import dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;
import se.llbit.log.Log;

import java.util.HashMap;
import java.util.Map;

/**
 * The Metal device and the render kernel, compiled from the same source as the OpenCL kernel
 * (see kernel/include/compat.h). Created on first use.
 */
public class MetalContext {
    public final long context;
    public final MetalSceneLoader sceneLoader;
    private final Map<String, Long> renderPipelines = new HashMap<>();

    private static volatile MetalContext instance;

    private MetalContext() {
        context = Metal.createContext();
        sceneLoader = new MetalSceneLoader(context);
        renderPipeline();
    }

    /** The render kernel compiled with the given defines (see AtmosphereData.kernelDefines), cached. */
    public long renderPipeline(String... defines) {
        return pipeline("render", defines);
    }

    /** The preview kernel compiled with the given defines, cached. */
    public long previewPipeline(String... defines) {
        return pipeline("preview", defines);
    }

    private synchronized long pipeline(String function, String... defines) {
        String key = function + ":" + String.join(",", defines);
        return renderPipelines.computeIfAbsent(key, k -> {
            long t0 = System.currentTimeMillis();
            // Safe (IEEE) math to match OpenCL, which the kernel was written and tested against.
            long pipeline = Metal.compile(context, KernelLoader.loadSource("kernel", "rayTracer.c", defines), function, false);
            Log.infof("ChunkyCL Metal: %s, kernel [%s] compiled in %d ms",
                    Metal.deviceName(context), k, System.currentTimeMillis() - t0);
            return pipeline;
        });
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
