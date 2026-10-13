package dev.thatredox.chunkynative.opencl.renderer.scene;


import static org.jocl.CL.*;

import dev.thatredox.chunkynative.common.scene.SkyData;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;

import se.llbit.chunky.renderer.scene.Scene;

public class ClSky implements AutoCloseable {
    public final ClMemory skyTexture;
    public final ClMemory skyIntensity;
    private final ClContext context;

    public ClSky(Scene scene, ClContext context) {
        this.context = context;
        SkyData sky = new SkyData(scene);

        this.skyIntensity = new ClMemory(clCreateBuffer(context.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float,
                Pointer.to(new float[] {sky.intensity}), null));

        cl_image_format fmt = new cl_image_format();
        fmt.image_channel_data_type = CL_UNORM_INT8;
        fmt.image_channel_order = CL_RGBA;

        cl_image_desc desc = new cl_image_desc();
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = sky.resolution;
        desc.image_height = sky.resolution;

        this.skyTexture = new ClMemory(clCreateImage(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                fmt, desc, Pointer.to(sky.texture), null));
    }

    @Override
    public void close() {
        skyTexture.close();
        skyIntensity.close();
    }
}
