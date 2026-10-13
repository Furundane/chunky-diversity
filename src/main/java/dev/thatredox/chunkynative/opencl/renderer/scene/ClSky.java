package dev.thatredox.chunkynative.opencl.renderer.scene;


import static org.jocl.CL.*;

import dev.thatredox.chunkynative.common.scene.SkyData;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;

import se.llbit.chunky.renderer.scene.Scene;

public class ClSky implements AutoCloseable {
    public final ClMemory skyTexture;
    public final ClMemory skyLight;
    private final ClContext context;

    public ClSky(Scene scene, ClContext context) {
        this.context = context;
        SkyData sky = new SkyData(scene);

        this.skyLight = new ClMemory(clCreateBuffer(context.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) Sizeof.cl_float * sky.skyLight.length,
                Pointer.to(sky.skyLight), null));

        this.skyTexture = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * sky.texture.length, Pointer.to(sky.texture), null));
    }

    @Override
    public void close() {
        skyTexture.close();
        skyLight.close();
    }
}
