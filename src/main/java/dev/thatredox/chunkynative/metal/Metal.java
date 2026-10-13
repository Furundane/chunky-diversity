package dev.thatredox.chunkynative.metal;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Thin JNI binding to Metal (src/main/native/metal_bridge.m). Handles are retained Objective-C
 * objects passed around as longs; free each one with {@link #release}.
 *
 * <p>The native library is bundled in the jar for macOS. {@code -Dchunkymetal.library=<path>}
 * loads it from a file instead.
 */
public final class Metal {
    private static final String LIBRARY_RESOURCE = "native/libchunkymetal.dylib";

    static {
        String override = System.getProperty("chunkymetal.library");
        if (override != null) {
            System.load(override);
        } else {
            System.load(extractLibrary().toString());
        }
    }

    private Metal() {}

    private static Path extractLibrary() {
        try (InputStream in = Metal.class.getClassLoader().getResourceAsStream(LIBRARY_RESOURCE)) {
            if (in == null) {
                throw new UnsatisfiedLinkError("Metal bridge library not found in the jar (only built on macOS)");
            }
            Path file = Files.createTempFile("chunkymetal", ".dylib");
            file.toFile().deleteOnExit();
            Files.copy(in, file, StandardCopyOption.REPLACE_EXISTING);
            return file;
        } catch (IOException e) {
            UnsatisfiedLinkError error = new UnsatisfiedLinkError("Could not extract the Metal bridge library");
            error.initCause(e);
            throw error;
        }
    }

    /** Creates a context (default device and one command queue). */
    public static native long createContext();

    public static native String deviceName(long context);

    /**
     * Compiles Metal Shading Language source at runtime and returns a compute pipeline for one
     * kernel function. Throws RuntimeException with the compiler log on failure.
     *
     * @param fastMath false for IEEE-conformant math, matching OpenCL's default
     */
    public static native long compile(long context, String source, String function, boolean fastMath);

    /** Allocates a zero-filled buffer in shared (CPU and GPU visible) memory. */
    public static native long newBuffer(long context, long bytes);

    private static native ByteBuffer contents0(long buffer);

    /**
     * Direct view of a buffer's memory, in native byte order. No copy: on Apple silicon the CPU
     * and GPU share memory. Valid until the buffer is released. Only touch it while no kernel
     * using the buffer is running.
     */
    public static ByteBuffer contents(long buffer) {
        return contents0(buffer).order(ByteOrder.nativeOrder());
    }

    /** Creates an RGBA8 (unsigned normalized) 2D texture array. */
    public static native long newTexture2DArray(long context, int width, int height, int layers);

    /** Creates an RGBA8 (unsigned normalized) 2D texture. */
    public static native long newTexture2D(long context, int width, int height);

    /** Writes tightly packed RGBA8 pixels into a region of one texture layer. */
    public static native void replaceRegion(long texture, int x, int y, int layer, int width, int height, byte[] rgba);

    /**
     * Runs a kernel over {@code threads} threads and waits for it to finish. Buffers and textures
     * are bound at indices 0..n-1 each, in the order the kernel declares them.
     *
     * @return GPU execution time in nanoseconds
     */
    public static native long dispatch(long context, long pipeline, long[] buffers, long[] textures, int threads);

    public static native void release(long handle);
}
