package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.common.export.Packer;
import dev.thatredox.chunkynative.common.export.ResourcePalette;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import it.unimi.dsi.fastutil.ints.IntArrayList;

/** Packed resources in one int buffer, uploaded on first use. Metal version of ClPackedResourcePalette. */
public class MetalPackedResourcePalette<T extends Packer> implements ResourcePalette<T> {
    protected MetalBuffer buffer = null;
    protected IntArrayList palette = new IntArrayList();
    protected final long context;

    public MetalPackedResourcePalette(long context) {
        this.context = context;
    }

    @Override
    public int put(T resource) {
        if (buffer != null) throw new IllegalStateException("Attempted to modify a locked palette.");
        int ptr = palette.size();
        palette.addAll(resource.pack());
        return ptr;
    }

    public MetalBuffer get() {
        if (buffer == null) {
            buffer = MetalBuffer.ofInts(context, palette);
        }
        return buffer;
    }

    @Override
    public void release() {
        if (buffer != null) buffer.close();
    }
}
