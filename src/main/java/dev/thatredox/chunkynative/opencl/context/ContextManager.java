package dev.thatredox.chunkynative.opencl.context;

import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import org.jocl.CLException;
import org.jocl.cl_program;
import se.llbit.log.Log;

import java.util.HashMap;
import java.util.Map;

public class ContextManager {
    public final Device device;
    public final ClContext context;
    public final Tonemap tonemap;
    public final Renderer renderer;

    public final ClSceneLoader sceneLoader;

    private static volatile ContextManager instance = new ContextManager(Device.getPreferredDevice());

    private ContextManager(Device device) {
        this.device = device;
        this.context = new ClContext(device);
        this.tonemap = new Tonemap(context);
        this.renderer = new Renderer(context);
        this.sceneLoader = new ClSceneLoader(context);
    }

    public static ContextManager get() {
        return instance;
    }

    public static synchronized void setDevice(Device device) {
        try {
            instance = new ContextManager(device);
        } catch (CLException e) {
            Log.error("Failed to set device", e);
        }
    }

    public static synchronized void reload() {
        setDevice(instance.device);
    }

    public static class Tonemap {
        public final cl_program simpleFilter;

        private Tonemap(ClContext context) {
            this.simpleFilter = KernelLoader.loadProgram(context, "tonemap", "post_processing_filter.c");
        }
    }

    public static class Renderer {
        private final ClContext context;
        private final Map<String, cl_program> variants = new HashMap<>();

        private Renderer(ClContext context) {
            this.context = context;
            // Build the plain variant now, so a broken kernel shows up when the plugin loads.
            get();
        }

        /** The render kernel program built with the given defines (see AtmosphereData.kernelDefines). */
        public synchronized cl_program get(String... defines) {
            return variants.computeIfAbsent(String.join(",", defines),
                    key -> context.loadProgram(KernelLoader.loadSource("kernel", "rayTracer.c", defines)));
        }
    }
}
