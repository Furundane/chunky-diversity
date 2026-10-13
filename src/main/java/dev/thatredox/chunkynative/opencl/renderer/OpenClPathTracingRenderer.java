package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.common.renderer.GpuPathTracingRenderer;
import dev.thatredox.chunkynative.common.scene.AtmosphereData;
import dev.thatredox.chunkynative.common.scene.CameraData;
import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;

import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.concurrent.locks.Lock;

public class OpenClPathTracingRenderer extends GpuPathTracingRenderer {

    @Override
    public String getId() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getName() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getDescription() {
        return "ChunkyClRenderer";
    }

    @Override
    protected RenderJob createJob(Scene scene, float[] passBuffer) {
        return new ClRenderJob(scene, passBuffer);
    }

    private static class ClRenderJob implements RenderJob {
        private final ContextManager context = ContextManager.get();
        private final ClSceneLoader sceneLoader = context.sceneLoader;
        private final cl_kernel kernel;
        private final ClCamera camera;
        private final ClMemory buffer;
        private final ClMemory randomSeed;
        private final ClMemory bufferSpp;
        private final ClIntBuffer clCanvasConfig;
        private final ClIntBuffer clRayDepth;
        private final ClWater water;
        private final ClIntBuffer atmosphere;
        private final int pixels;

        ClRenderJob(Scene scene, float[] passBuffer) {
            // Ensure the scene is loaded
            sceneLoader.ensureLoad(scene);

            // Load the kernel
            kernel = clCreateKernel(context.renderer.get(AtmosphereData.kernelDefines(scene)), "render", null);

            // Generate the camera
            camera = new ClCamera(scene, context.context);
            buffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer), null));
            randomSeed = new ClMemory(
                    clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_int, null, null));
            bufferSpp = new ClMemory(
                    clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_int, null, null));
            clCanvasConfig = new ClIntBuffer(CameraData.canvasConfig(scene), context.context);
            clRayDepth = new ClIntBuffer(scene.getRayDepth(), context.context);
            water = new ClWater(scene, context.context);
            atmosphere = new ClIntBuffer(AtmosphereData.pack(scene), context.context);
            pixels = passBuffer.length / 3;
        }

        @Override
        public boolean needsCameraGeneration() {
            return camera.needGenerate;
        }

        @Override
        public void generateCamera(Lock renderLock) {
            camera.generate(renderLock, true);
        }

        // -Dchunkycl.profilePasses=true prints the time per pass (enqueue to finish) when the job ends.
        private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");
        private long passNanos = 0;
        private int passes = 0;

        @Override
        public void renderPass(int seed, int spp) {
            cl_event[] renderEvent = { new cl_event() };
            long start = System.nanoTime();

            clEnqueueWriteBuffer(context.context.queue, randomSeed.get(), CL_TRUE, 0, Sizeof.cl_int,
                    Pointer.to(new int[]{seed}), 0, null, null);
            clEnqueueWriteBuffer(context.context.queue, bufferSpp.get(), CL_TRUE, 0, Sizeof.cl_int,
                    Pointer.to(new int[]{spp}), 0, null, null);

            int argIndex = 0;
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.projectorType.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.cameraSettings.get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeDepth().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeDepth().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeData().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBlockPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getQuadPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getAabbPalette().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWorldBvh().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getActorBvh().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTrigPalette().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTexturePalette().getAtlas()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getMaterialPalette().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyTexture.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyLight.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSun().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(water.settings.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(atmosphere.get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(randomSeed.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(bufferSpp.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clCanvasConfig.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clRayDepth.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));
            clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                    new long[]{pixels}, null, 0, null,
                    renderEvent[0]);
            clWaitForEvents(1, renderEvent);
            clReleaseEvent(renderEvent[0]);
            passNanos += System.nanoTime() - start;
            passes++;
        }

        @Override
        public void readBuffer(float[] passBuffer) {
            clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                    (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer),
                    0, null, null);
        }

        @Override
        public void close() {
            camera.close();
            buffer.close();
            randomSeed.close();
            bufferSpp.close();
            clCanvasConfig.close();
            clRayDepth.close();
            water.close();
            atmosphere.close();
            clReleaseKernel(kernel);
            if (PROFILE && passes > 0) {
                System.out.printf(java.util.Locale.ROOT, "CL_PROFILE passes=%d msPerPass=%.2f%n", passes, passNanos / 1e6 / passes);
            }
        }
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        ContextManager.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
    }
}
