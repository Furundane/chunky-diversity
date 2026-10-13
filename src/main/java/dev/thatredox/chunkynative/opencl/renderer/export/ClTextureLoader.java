package dev.thatredox.chunkynative.opencl.renderer.export;

import dev.thatredox.chunkynative.common.export.texture.AtlasTextureLoader;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;

import java.util.List;

import static org.jocl.CL.*;

public class ClTextureLoader extends AtlasTextureLoader implements AutoCloseable {
    private ClMemory texture;
    private final ClContext context;

    public ClTextureLoader(ClContext context) {
        this.context = context;
    }

    public cl_mem getAtlas() {
        return texture.get();
    }

    @Override
    public void close() {
        texture.close();
    }

    @Override
    protected void upload(int layers, List<AtlasTexture> texs) {
        cl_image_format fmt = new cl_image_format();
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_UNORM_INT8;

        cl_image_desc desc = new cl_image_desc();
        desc.image_width = ATLAS_SIZE;
        desc.image_height = ATLAS_SIZE;
        desc.image_array_size = layers;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;

        texture = new ClMemory(
                clCreateImage(context.context, CL_MEM_READ_ONLY, fmt, desc, null, null));

        for (AtlasTexture tex : texs) {
            clEnqueueWriteImage(context.queue, texture.get(), CL_TRUE,
                    new long[] {tex.getX()* 16L, tex.getY()* 16L, tex.getD()},
                    new long[] {tex.getWidth(), tex.getHeight(), 1},
                    0, 0, Pointer.to(tex.getTexture()),
                    0, null, null
            );
        }
    }
}
