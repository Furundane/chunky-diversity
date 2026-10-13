package dev.thatredox.chunkynative.metal.util;

import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.util.NativeCleaner;

/** An RGBA8 Metal texture. Released on {@link #close()}, or when garbage collected. */
public class MetalTexture implements AutoCloseable {
    public final long handle;
    private final NativeCleaner.Cleaner cleaner;

    private MetalTexture(long handle) {
        this.handle = handle;
        this.cleaner = NativeCleaner.INSTANCE.register(this, () -> Metal.release(handle));
    }

    public static MetalTexture create2D(long context, int width, int height, byte[] rgba) {
        MetalTexture texture = new MetalTexture(Metal.newTexture2D(context, width, height));
        Metal.replaceRegion(texture.handle, 0, 0, 0, width, height, rgba);
        return texture;
    }

    public static MetalTexture create2DArray(long context, int width, int height, int layers) {
        return new MetalTexture(Metal.newTexture2DArray(context, width, height, layers));
    }

    public void write(int x, int y, int layer, int width, int height, byte[] rgba) {
        Metal.replaceRegion(handle, x, y, layer, width, height, rgba);
    }

    @Override
    public void close() {
        cleaner.clean();
    }
}
