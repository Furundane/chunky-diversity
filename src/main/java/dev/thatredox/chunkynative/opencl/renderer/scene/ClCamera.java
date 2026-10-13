package dev.thatredox.chunkynative.opencl.renderer.scene;

import dev.thatredox.chunkynative.common.scene.CameraData;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.concurrent.locks.Lock;

import static org.jocl.CL.*;

public class ClCamera implements AutoCloseable {
    public ClMemory projectorType;
    public ClMemory cameraSettings;
    public final boolean needGenerate;

    private final Scene scene;
    private final ClContext context;


    public ClCamera(Scene scene, ClContext context) {
        this.scene = scene;
        this.context = context;
        CameraData camera = new CameraData(scene);
        needGenerate = camera.needsRayGeneration();

        projectorType = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int, Pointer.to(new int[] {camera.projectorType}), null));

        if (needGenerate) {
            cameraSettings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY,
                    (long) Sizeof.cl_float * CameraData.rayBufferFloats(scene), null, null));
        } else {
            cameraSettings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * camera.settings.length, Pointer.to(camera.settings), null));
        }
    }

    public void generate(Lock renderLock, boolean jitter) {
        if (!needGenerate) return;

        float[] rays = CameraData.generateRays(scene, jitter);

        if (renderLock != null) renderLock.lock();
        clEnqueueWriteBuffer(context.queue, this.cameraSettings.get(), CL_TRUE, 0,
                (long) Sizeof.cl_float * rays.length, Pointer.to(rays), 0,
                null, null);
        if (renderLock != null) renderLock.unlock();
    }

    @Override
    public void close() {
        this.projectorType.close();
        this.cameraSettings.close();
    }
}
