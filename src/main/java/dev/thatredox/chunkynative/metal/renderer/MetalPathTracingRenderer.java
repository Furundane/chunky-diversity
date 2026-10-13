package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.common.renderer.GpuPathTracingRenderer;
import dev.thatredox.chunkynative.common.scene.AtmosphereData;
import dev.thatredox.chunkynative.common.scene.CameraData;
import dev.thatredox.chunkynative.common.scene.WaterData;
import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.concurrent.locks.Lock;

/** GPU path tracer on Metal, running the same kernel as the OpenCL renderer. */
public class MetalPathTracingRenderer extends GpuPathTracingRenderer {
    public static final String ID = "ChunkyMetalRenderer";

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getName() {
        return "ChunkyCL (Metal)";
    }

    @Override
    public String getDescription() {
        return "GPU path tracer using Metal";
    }

    @Override
    protected RenderJob createJob(Scene scene, float[] passBuffer) {
        return new MetalRenderJob(scene, passBuffer);
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        MetalContext.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
    }

    private static class MetalRenderJob implements RenderJob {
        private final MetalContext metal = MetalContext.get();
        private final MetalSceneLoader sceneLoader = metal.sceneLoader;
        private final Scene scene;
        private final boolean needGenerate;
        private final MetalBuffer projectorType;
        private final MetalBuffer cameraSettings;
        private final MetalBuffer buffer;
        private final MetalBuffer randomSeed;
        private final MetalBuffer bufferSpp;
        private final MetalBuffer canvasConfig;
        private final MetalBuffer rayDepth;
        private final MetalBuffer water;
        private final MetalBuffer atmosphere;
        private final int pixels;
        private final long pipeline;
        private final BlockTracing tracing;
        // -Dchunkymetal.profile=true prints the GPU time per pass when the job ends.
        private static final boolean PROFILE = Boolean.getBoolean("chunkymetal.profile");
        private long gpuNanos = 0;
        private int passes = 0;

        MetalRenderJob(Scene scene, float[] passBuffer) {
            this.scene = scene;
            sceneLoader.ensureLoad(scene);
            tracing = new BlockTracing(sceneLoader);
            pipeline = metal.renderPipeline(tracing.defines(AtmosphereData.kernelDefines(scene)));
            tracing.bind(metal.context, pipeline, sceneLoader);

            long context = metal.context;
            CameraData camera = new CameraData(scene);
            needGenerate = camera.needsRayGeneration();
            projectorType = MetalBuffer.ofInt(context, camera.projectorType);
            cameraSettings = needGenerate
                    ? new MetalBuffer(context, 4L * CameraData.rayBufferFloats(scene))
                    : MetalBuffer.ofFloats(context, camera.settings);
            buffer = MetalBuffer.ofFloats(context, passBuffer);
            randomSeed = MetalBuffer.ofInt(context, 0);
            bufferSpp = MetalBuffer.ofInt(context, 0);
            canvasConfig = MetalBuffer.ofInts(context, CameraData.canvasConfig(scene));
            rayDepth = MetalBuffer.ofInt(context, scene.getRayDepth());
            water = MetalBuffer.ofFloats(context, WaterData.pack(scene));
            atmosphere = MetalBuffer.ofInts(context, AtmosphereData.pack(scene));
            pixels = passBuffer.length / 3;
        }

        @Override
        public boolean needsCameraGeneration() {
            return needGenerate;
        }

        @Override
        public void generateCamera(Lock renderLock) {
            if (!needGenerate) return;
            float[] rays = CameraData.generateRays(scene, true);
            renderLock.lock();
            try {
                cameraSettings.writeFloats(0, rays, 0, rays.length);
            } finally {
                renderLock.unlock();
            }
        }

        @Override
        public void renderPass(int seed, int spp) {
            randomSeed.writeInt(0, seed);
            bufferSpp.writeInt(0, spp);

            // Buffers and textures in the order the render kernel declares them (rayTracer.c).
            long[] buffers = tracing.buffers(new long[] {
                    projectorType.handle,
                    cameraSettings.handle,

                    sceneLoader.getOctreeDepth().handle,
                    sceneLoader.getOctreeData().handle,
            }, new long[] {
                    sceneLoader.getWaterOctreeDepth().handle,
                    sceneLoader.getWaterOctreeData().handle,

                    sceneLoader.getBlockPalette().handle,
                    sceneLoader.getQuadPalette().handle,
                    sceneLoader.getAabbPalette().handle,

                    sceneLoader.getWorldBvh().handle,
                    sceneLoader.getActorBvh().handle,
                    sceneLoader.getTrigPalette().handle,

                    sceneLoader.getMaterialPalette().handle,
                    sceneLoader.getSkyTexture().handle,

                    sceneLoader.getSkyLight().handle,
                    sceneLoader.getSun().handle,
                    water.handle,
                    atmosphere.handle,

                    randomSeed.handle,
                    bufferSpp.handle,
                    canvasConfig.handle,
                    rayDepth.handle,
                    buffer.handle,
            });
            long[] textures = {
                    sceneLoader.getTexturePalette().getAtlas().handle,
            };
            gpuNanos += Metal.dispatch(metal.context, pipeline, buffers, textures, pixels);
            passes++;
        }

        @Override
        public void readBuffer(float[] passBuffer) {
            buffer.readFloats(passBuffer);
        }

        @Override
        public void close() {
            projectorType.close();
            cameraSettings.close();
            buffer.close();
            randomSeed.close();
            bufferSpp.close();
            canvasConfig.close();
            rayDepth.close();
            water.close();
            atmosphere.close();
            tracing.close();
            if (PROFILE && passes > 0) {
                System.out.printf(java.util.Locale.ROOT, "METAL_PROFILE passes=%d gpuMsPerPass=%.2f%n", passes, gpuNanos / 1e6 / passes);
            }
        }
    }
}
