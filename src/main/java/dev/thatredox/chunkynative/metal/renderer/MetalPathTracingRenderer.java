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
        return MetalContext.wavefront() ? new WavefrontJob(scene, passBuffer) : new MetalRenderJob(scene, passBuffer);
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
        // -Dchunkycl.profilePasses=true prints the GPU time per pass when the job ends.
        private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");
        private long gpuNanos = 0;
        private int passes = 0;

        MetalRenderJob(Scene scene, float[] passBuffer) {
            this.scene = scene;
            sceneLoader.ensureLoad(scene);
            pipeline = metal.renderPipeline(AtmosphereData.kernelDefines(scene));

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
            long[] buffers = {
                    projectorType.handle,
                    cameraSettings.handle,

                    sceneLoader.getOctreeDepth().handle,
                    sceneLoader.getOctreeData().handle,
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
            };
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
            if (PROFILE && passes > 0) {
                System.out.printf(java.util.Locale.ROOT, "METAL_PROFILE passes=%d gpuMsPerPass=%.2f%n", passes, gpuNanos / 1e6 / passes);
            }
        }
    }

    /**
     * The wavefront render loop (kernel/include/wavefront.c): per pass and wave of up to
     * {@link #WAVE_SIZE} pixels, wf_generate, then wf_intersect, wf_shade and wf_shadow per bounce,
     * then wf_finish, all in one command buffer.
     */
    private static class WavefrontJob implements RenderJob {
        /** Paths traced at once; the path state takes about 190 bytes per path. */
        private static final int WAVE_SIZE = 1 << 20;
        private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");

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
        private final MetalBuffer rayDepthBuffer;
        private final MetalBuffer water;
        private final MetalBuffer atmosphere;
        private final MetalBuffer pathF, pathI, hitF, hitI, shadowF, shadowI;
        /** Per wave and bounce: first pixel, number of paths, bounce. */
        private final MetalBuffer[][] waves;
        private final int pixels;
        private final int rayDepth;
        private final long[] pipelines;
        private long gpuNanos = 0;
        private int passes = 0;

        WavefrontJob(Scene scene, float[] passBuffer) {
            this.scene = scene;
            sceneLoader.ensureLoad(scene);
            pipelines = metal.wavefrontPipelines(AtmosphereData.kernelDefines(scene));

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
            rayDepth = scene.getRayDepth();
            rayDepthBuffer = MetalBuffer.ofInt(context, rayDepth);
            water = MetalBuffer.ofFloats(context, WaterData.pack(scene));
            atmosphere = MetalBuffer.ofInts(context, AtmosphereData.pack(scene));
            pixels = passBuffer.length / 3;

            long paths = Math.min(pixels, WAVE_SIZE);
            pathF = new MetalBuffer(context, paths * 12 * 4);
            pathI = new MetalBuffer(context, paths * 4 * 4);
            hitF = new MetalBuffer(context, paths * 14 * 4);
            hitI = new MetalBuffer(context, paths * 5 * 4);
            shadowF = new MetalBuffer(context, paths * 9 * 4);
            shadowI = new MetalBuffer(context, paths * 3 * 4);

            int waveCount = (pixels + WAVE_SIZE - 1) / WAVE_SIZE;
            waves = new MetalBuffer[waveCount][rayDepth + 1];
            for (int w = 0; w < waveCount; w++) {
                int first = w * WAVE_SIZE;
                int count = Math.min(WAVE_SIZE, pixels - first);
                for (int d = 0; d <= rayDepth; d++) {
                    waves[w][d] = MetalBuffer.ofInts(context, new int[] {first, count, d});
                }
            }
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

        private long[] buffers(MetalBuffer wave) {
            // In the order the wavefront kernels declare them (WF_PARAMS in wavefront.c).
            return new long[] {
                    projectorType.handle,
                    cameraSettings.handle,

                    sceneLoader.getOctreeDepth().handle,
                    sceneLoader.getOctreeData().handle,
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
                    rayDepthBuffer.handle,
                    buffer.handle,

                    pathF.handle,
                    pathI.handle,
                    hitF.handle,
                    hitI.handle,
                    shadowF.handle,
                    shadowI.handle,
                    wave.handle,
            };
        }

        @Override
        public void renderPass(int seed, int spp) {
            randomSeed.writeInt(0, seed);
            bufferSpp.writeInt(0, spp);

            long[] textures = {sceneLoader.getTexturePalette().getAtlas().handle};
            long generate = pipelines[0], intersect = pipelines[1], shade = pipelines[2], shadow = pipelines[3], finish = pipelines[4];
            long batch = Metal.beginBatch(metal.context);
            for (int w = 0; w < waves.length; w++) {
                int count = Math.min(WAVE_SIZE, pixels - w * WAVE_SIZE);
                Metal.addDispatch(batch, generate, buffers(waves[w][0]), textures, count);
                for (int d = 0; d < rayDepth; d++) {
                    long[] buffers = buffers(waves[w][d]);
                    Metal.addDispatch(batch, intersect, buffers, textures, count);
                    Metal.addDispatch(batch, shade, buffers, textures, count);
                    Metal.addDispatch(batch, shadow, buffers, textures, count);
                }
                Metal.addDispatch(batch, finish, buffers(waves[w][0]), textures, count);
            }
            gpuNanos += Metal.runBatch(batch);
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
            rayDepthBuffer.close();
            water.close();
            atmosphere.close();
            pathF.close();
            pathI.close();
            hitF.close();
            hitI.close();
            shadowF.close();
            shadowI.close();
            for (MetalBuffer[] wave : waves) {
                for (MetalBuffer b : wave) b.close();
            }
            if (PROFILE && passes > 0) {
                System.out.printf(java.util.Locale.ROOT, "METAL_PROFILE wavefront passes=%d gpuMsPerPass=%.2f%n", passes, gpuNanos / 1e6 / passes);
            }
        }
    }
}
