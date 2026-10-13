package dev.thatredox.chunkynative.opencl.renderer.scene;

import dev.thatredox.chunkynative.common.scene.WaterData;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import se.llbit.chunky.renderer.scene.Scene;

import static org.jocl.CL.*;

/** Water settings for the kernel, see {@link WaterData}. */
public class ClWater implements AutoCloseable {
    public final ClMemory settings;

    public ClWater(Scene scene, ClContext context) {
        float[] data = WaterData.pack(scene);
        settings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * data.length, Pointer.to(data), null));
    }

    @Override
    public void close() {
        settings.close();
    }
}
