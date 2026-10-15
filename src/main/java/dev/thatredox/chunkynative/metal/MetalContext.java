package dev.thatredox.chunkynative.metal;

import dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;
import se.llbit.log.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The Metal device and the render kernel, compiled from the same source as the OpenCL kernel
 * (see kernel/include/compat.h). Created on first use.
 */
public class MetalContext {
    public final long context;
    public final MetalSceneLoader sceneLoader;
    private final Map<String, Long> renderPipelines = new HashMap<>();
    private final Map<String, long[]> wavefrontPipelines = new HashMap<>();

    /** Kernel define that traces the world octree with hardware ray tracing (hwrt.h). */
    public static final String HWRT_DEFINE = "CHUNKYCL_HWRT";

    private static volatile MetalContext instance;

    private MetalContext() {
        context = Metal.createContext();
        sceneLoader = new MetalSceneLoader(context);
        renderPipeline();
    }

    /** The defines with {@link #HWRT_DEFINE} added when a block acceleration structure is in use. */
    public static String[] withAcceleration(String[] defines, long blockAccel) {
        if (blockAccel == 0) return defines;
        String[] result = Arrays.copyOf(defines, defines.length + 1);
        result[defines.length] = HWRT_DEFINE;
        return result;
    }

    /** The render kernel compiled with the given defines (see AtmosphereData.kernelDefines), cached. */
    public long renderPipeline(String... defines) {
        return pipeline("render", defines);
    }

    /** Kernel functions of the wavefront render loop (kernel/include/wavefront.c), in this order. */
    public static final String[] WAVEFRONT_FUNCTIONS = {
            "wf_generate", "wf_intersect", "wf_shade", "wf_shadow", "wf_finish", "wf_prepare", "wf_prepareShadow",
            "wf_traceProbe", "wf_begin", "wf_prepareTrace", "wf_trace", "wf_resolve", "wf_intersectRest", "wf_world", "wf_traceWater", "wf_resolveWater"};

    /** Whether the Metal renderer runs the wavefront render loop (see {@link MetalSettings#wavefront}). */
    public static boolean wavefront() {
        return MetalSettings.wavefront();
    }

    /** The wavefront kernels ({@link #WAVEFRONT_FUNCTIONS}) compiled with the given defines, cached. */
    public synchronized long[] wavefrontPipelines(String... baseDefines) {
        // -Dchunkymetal.defines=A,B adds kernel defines (for diagnostics).
        List<String> all = new ArrayList<>(Arrays.asList(baseDefines));
        String extra = System.getProperty("chunkymetal.defines", "");
        if (!extra.isEmpty()) all.addAll(Arrays.asList(extra.split(",")));
        String[] defines = all.toArray(new String[0]);
        String key = String.join(",", defines);
        return wavefrontPipelines.computeIfAbsent(key, k -> {
            long t0 = System.currentTimeMillis();
            long[] pipelines = Metal.compileFunctions(context,
                    KernelLoader.loadSource("kernel", "wavefront.c", defines), WAVEFRONT_FUNCTIONS, false,
                    Arrays.asList(defines).contains(HWRT_DEFINE) ? "HwRt_firstBlock" : null);
            Log.infof("ChunkyCL Metal: %s, wavefront kernels [%s] compiled in %d ms",
                    Metal.deviceName(context), k, System.currentTimeMillis() - t0);
            return pipelines;
        });
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
            String source = KernelLoader.loadSource("kernel", "rayTracer.c", defines);
            long pipeline = Arrays.asList(defines).contains(HWRT_DEFINE)
                    ? Metal.compileRt(context, source, function, false, "HwRt_firstBlock")
                    : Metal.compile(context, source, function, false);
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
