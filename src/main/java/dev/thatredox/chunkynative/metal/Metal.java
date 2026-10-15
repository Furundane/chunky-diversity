package dev.thatredox.chunkynative.metal;

import java.io.IOException;
import java.io.InputStream;
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

    /** The largest buffer the device can allocate, in bytes. */
    public static native long maxBufferLength(long context);

    /*
     * Copy between Java arrays and a buffer, at a byte offset into the buffer (which can be larger
     * than 2 GB). Only call these while no kernel using the buffer is running.
     */
    public static native void writeInts(long buffer, long byteOffset, int[] data, int offset, int length);

    public static native void writeFloats(long buffer, long byteOffset, float[] data, int offset, int length);

    public static native void readFloats(long buffer, long byteOffset, float[] data, int offset, int length);

    public static native void readInts(long buffer, long byteOffset, int[] data, int offset, int length);

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

    /**
     * Like {@link #compile}, for several kernel functions of the same source; with an intersection
     * function (or null), like {@link #compileRt}.
     */
    public static native long[] compileFunctions(long context, String source, String[] functions, boolean fastMath, String intersectionFunction);

    /*
     * Batches: dispatches recorded with addDispatch run one after the other (each sees the results
     * of the previous ones) in a single command buffer when runBatch is called. runBatch waits for
     * them, returns the GPU time in nanoseconds and frees the batch.
     */
    public static native long beginBatch(long context);

    public static native void addDispatch(long batch, long pipeline, long[] buffers, long[] textures, int threads);

    public static native long runBatch(long batch);

    /**
     * A batch on another command queue (0 is the queue of {@link #beginBatch}). Batches on different
     * queues, run together with {@link #runBatches}, may execute at the same time on the GPU, so
     * they must not depend on each other.
     */
    public static native long beginBatchOn(long context, int queue);

    /** Runs several batches together and frees them; returns the GPU time from the first start to the last end (ns). */
    public static native long runBatches(long[] batches);

    /*
     * Indirect dispatches: the number of threadgroups of groupSize threads is read on the GPU from
     * the buffer `indirect` at byteOffset (3 uints: x, y, z), as written by an earlier dispatch.
     */
    public static native void addDispatchIndirect(long batch, long pipeline, long[] buffers, long[] textures, int groupSize,
                                                  long indirect, long byteOffset);

    /** Like {@link #dispatch}, with the threadgroup count read from a buffer (see addDispatchIndirect). */
    public static native long dispatchIndirect(long context, long pipeline, long[] buffers, long[] textures, int groupSize,
                                               long indirect, long byteOffset);

    /*
     * Hardware ray tracing of the world octree (kernel side: kernel/include/hwrt.h). Pipelines from
     * compileRt, block acceleration structures and function tables are passed to dispatch like
     * buffers, at the kernel's index of the acceleration structure and the function table.
     */

    /** Whether the GPU supports Metal ray tracing (fast in hardware on Apple M3 and newer). */
    public static native boolean supportsRaytracing(long context);

    /** Like {@link #compile}, linked with an intersection function for {@link #newFunctionTable}. */
    public static native long compileRt(long context, String source, String function, boolean fastMath, String intersectionFunction);

    /**
     * Builds bounding boxes and an acceleration structure for a packed world octree (as uploaded for
     * the kernel). Throws RuntimeException with the reason when it can't, e.g. not enough memory.
     *
     * @param sectionShift 2, 3 or 4 for 4^3, 8^3 or 16^3 sections, 0 for one box per octree leaf
     * @param budgetBytes  memory limit for the boxes and the build, 0 for half of the GPU's
     *                     recommended working set minus what is already allocated
     */
    public static native long buildBlockAccel(long context, long octree, int depth, long blockPalette, int sectionShift, long budgetBytes);

    /** Box count, acceleration structure bytes, box data bytes and build time (ns) of a block acceleration structure. */
    public static native long[] blockAccelInfo(long accel);

    /**
     * Intersection function table of an rt pipeline for a block acceleration structure: its box data,
     * then sceneBuffers, then the atlas texture in an argument buffer (0: none), at the function's
     * buffer indices in that order.
     */
    public static native long newFunctionTable(long context, long rtPipeline, long accel, long[] sceneBuffers, long atlas);

    /** Takes another reference to a handle (each reference is freed with {@link #release}). */
    public static native long retain(long handle);

    public static native void release(long handle);

    /**
     * A MetalFX temporal denoised scaler (macOS 26) from an input to an output size, with the textures
     * and kernels around it (see {@link #denoise}); 0 if MetalFX can't do this here. Freed with {@link #release}.
     */
    public static native long newDenoiser(long context, int inWidth, int inHeight, int outWidth, int outHeight);

    /**
     * Denoises and upscales one frame of the render kernel compiled with CHUNKYCL_AOV (color and aov
     * buffers at the input size) into ARGB pixels at the output size. params: the layout in
     * metal_bridge.m (denoiseSource); scalerParams: jitter x, y, reset history (0/1), world-to-view and
     * view-to-clip matrices (column-major). Returns the GPU time in nanoseconds.
     */
    public static native long denoise(long context, long denoiser, long color, long aov, long argb, float[] params, float[] scalerParams);
}
