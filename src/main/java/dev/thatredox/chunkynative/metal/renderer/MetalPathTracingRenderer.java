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
        private final BlockTracing tracing;
        // -Dchunkycl.profilePasses=true prints the GPU time per pass when the job ends.
        private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");
        private long gpuNanos = 0;
        private int passes = 0;

        MetalRenderJob(Scene scene, float[] passBuffer) {
            this.scene = scene;
            sceneLoader.ensureLoad(scene);
            tracing = new BlockTracing(sceneLoader);
            pipeline = metal.renderPipeline(tracing.defines(AtmosphereData.kernelDefines(scene)));
            tracing.bind(metal.context, pipeline);

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
            long[] buffers = tracing.buffers(pipeline, new long[] {
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

    /**
     * The wavefront render loop (kernel/include/wavefront.c): per pass and wave of up to
     * {@link #WAVE_SIZE} pixels, wf_generate, then wf_intersect, wf_shade and wf_shadow per bounce,
     * then wf_finish, all in one command buffer.
     */
    private static class WavefrontJob implements RenderJob {
        /** Paths traced at once; the path state takes about 190 bytes per path. */
        private static final int WAVE_SIZE = 1 << 20;
        private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");
        // -Dchunkymetal.wavefrontBreakdown=true: every dispatch in its own command buffer, to print
        // the GPU time of each kernel (slower; for profiling only).
        private static final boolean BREAKDOWN = Boolean.getBoolean("chunkymetal.wavefrontBreakdown");
        private final long[] kernelNanos = new long[MetalContext.WAVEFRONT_FUNCTIONS.length];
        // Kernels, in the order of MetalContext.WAVEFRONT_FUNCTIONS.
        private static final int GENERATE = 0, INTERSECT = 1, SHADE = 2, SHADOW = 3, FINISH = 4, PREPARE = 5, PREPARE_SHADOW = 6,
                TRACE_PROBE = 7, BEGIN = 8, PREPARE_TRACE = 9, TRACE = 10, RESOLVE = 11, INTERSECT_REST = 12, WORLD = 13,
                TRACE_WATER = 14, RESOLVE_WATER = 15;
        /** How the world octree is traced with hardware ray tracing: "split" (rounds) or "fused" (wf_world). */
        private static final String TRACE_MODE = System.getProperty("chunkymetal.traceMode", "split");
        /**
         * Rounds of wf_trace and wf_resolve per bounce when the world octree is traced with hardware
         * ray tracing; the last round finishes the remaining traces in wf_resolve.
         */
        private static final int TRACE_ROUNDS = Integer.getInteger("chunkymetal.traceRounds", 3);
        /** Octree leaves wf_resolve marches from each block found before another hardware query. */
        private static final int RESOLVE_STEPS = Integer.getInteger("chunkymetal.resolveSteps", 8);
        // -Dchunkymetal.traceProbe=true runs wf_traceProbe before wf_intersect (diagnostic).
        private static final boolean TRACE_PROBE_ON = Boolean.getBoolean("chunkymetal.traceProbe");
        /** Threadgroup size of the indirect dispatches (WF_GROUP in wavefront.c). */
        private static final int GROUP_SIZE = 64;
        /** Byte offsets of the indirect dispatch arguments in the control buffer (CONTROL_ARGS, CONTROL_SHADOW_ARGS). */
        private static final long ARGS = 4 * 4, SHADOW_ARGS = 8 * 4, TRACE_ARGS = 20 * 4;
        private static final boolean STATS = System.getProperty("chunkymetal.defines", "").contains("CHUNKYCL_HWRT_STATS");

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
        /** Path state (STATE_F floats, then STATE_I ints per path). */
        private final MetalBuffer state;
        /** Control block (queue lengths, dispatch arguments, statistics), then the queues. */
        private final MetalBuffer work;
        /**
         * Per wave, bounce and trace round: first pixel, number of paths, bounce, paths the state
         * buffers hold, input queue, input trace queue, last trace round, leaves wf_resolve marches.
         */
        private final MetalBuffer[][][] waves;
        private final int pixels;
        private final int rayDepth;
        private final long[] pipelines;
        private final BlockTracing tracing;
        private long gpuNanos = 0;
        private int passes = 0;

        WavefrontJob(Scene scene, float[] passBuffer) {
            this.scene = scene;
            sceneLoader.ensureLoad(scene);
            tracing = new BlockTracing(sceneLoader, true);
            pipelines = metal.wavefrontPipelines(tracing.defines(AtmosphereData.kernelDefines(scene)));
            for (long pipeline : pipelines) tracing.bind(metal.context, pipeline);

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
            // STATE_F + STATE_I, CONTROL_SIZE and five queues in wavefront.c
            state = new MetalBuffer(context, paths * (38 + 22) * 4);
            work = new MetalBuffer(context, (32 + paths * 5) * 4);

            int waveCount = (pixels + WAVE_SIZE - 1) / WAVE_SIZE;
            int rounds = Math.max(1, TRACE_ROUNDS);
            waves = new MetalBuffer[waveCount][rayDepth + 1][rounds];
            for (int w = 0; w < waveCount; w++) {
                int first = w * WAVE_SIZE;
                int count = Math.min(WAVE_SIZE, pixels - first);
                for (int d = 0; d <= rayDepth; d++) {
                    for (int r = 0; r < rounds; r++) {
                        waves[w][d][r] = MetalBuffer.ofInts(context,
                                new int[] {first, count, d, (int) paths, d % 2, r % 2, r == rounds - 1 ? 1 : 0, RESOLVE_STEPS});
                    }
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

        private long[] buffers(long pipeline, MetalBuffer wave) {
            // In the order the wavefront kernels declare them (WF_PARAMS in wavefront.c).
            return tracing.buffers(pipeline, new long[] {
                    projectorType.handle,
                    cameraSettings.handle,

                    sceneLoader.getOctreeDepth().handle,
                    sceneLoader.getOctreeData().handle,
            }, new long[] {
                    sceneLoader.getWaterOctreeDepth().handle,
                    sceneLoader.getWaterOctreeData().handle,
            }, new long[] {

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

                    state.handle,
                    work.handle,
                    wave.handle,
            });
        }

        /** Runs or records one dispatch of the pass. */
        private interface Dispatcher {
            void direct(int kernel, MetalBuffer wave, int threads);

            void indirect(int kernel, MetalBuffer wave, long argsOffset);
        }

        /**
         * One pass: per wave, generate (queues every path), per bounce prepare, the intersection of
         * the queued paths, shade, then prepareShadow and shadow over the shadow queue, then finish.
         * With hardware ray tracing the intersection runs in parts: begin, rounds of prepareTrace,
         * trace and resolve, then intersectRest.
         */
        private void pass(Dispatcher d) {
            boolean split = tracing.hardware() && !Boolean.getBoolean("chunkymetal.noTraceSplit");
            for (int w = 0; w < waves.length; w++) {
                int count = Math.min(WAVE_SIZE, pixels - w * WAVE_SIZE);
                d.direct(GENERATE, waves[w][0][0], count);
                for (int depth = 0; depth < rayDepth; depth++) {
                    MetalBuffer wave = waves[w][depth][0];
                    d.direct(PREPARE, wave, 1);
                    if (TRACE_PROBE_ON) d.indirect(TRACE_PROBE, wave, ARGS);
                    if (split) {
                        d.indirect(BEGIN, wave, ARGS);
                        if (TRACE_MODE.equals("fused")) {
                            d.indirect(WORLD, wave, ARGS);
                        } else {
                            for (MetalBuffer round : waves[w][depth]) {
                                d.direct(PREPARE_TRACE, round, 1);
                                d.indirect(TRACE, round, TRACE_ARGS);
                                d.indirect(RESOLVE, round, TRACE_ARGS);
                            }
                        }
                        d.indirect(INTERSECT_REST, wave, ARGS);
                        if (tracing.hardwareWater()) {
                            d.indirect(TRACE_WATER, wave, ARGS);
                            d.indirect(RESOLVE_WATER, wave, ARGS);
                        }
                    } else {
                        d.indirect(INTERSECT, wave, ARGS);
                    }
                    d.indirect(SHADE, wave, ARGS);
                    d.direct(PREPARE_SHADOW, wave, 1);
                    d.indirect(SHADOW, wave, SHADOW_ARGS);
                }
                d.direct(FINISH, waves[w][0][0], count);
            }
        }

        @Override
        public void renderPass(int seed, int spp) {
            randomSeed.writeInt(0, seed);
            bufferSpp.writeInt(0, spp);

            long[] textures = {sceneLoader.getTexturePalette().getAtlas().handle};
            if (BREAKDOWN) {
                pass(new Dispatcher() {
                    @Override
                    public void direct(int kernel, MetalBuffer wave, int threads) {
                        kernelNanos[kernel] += Metal.dispatch(metal.context, pipelines[kernel],
                                buffers(pipelines[kernel], wave), textures, threads);
                    }

                    @Override
                    public void indirect(int kernel, MetalBuffer wave, long argsOffset) {
                        kernelNanos[kernel] += Metal.dispatchIndirect(metal.context, pipelines[kernel],
                                buffers(pipelines[kernel], wave), textures, GROUP_SIZE, work.handle, argsOffset);
                    }
                });
                passes++;
                return;
            }
            long batch = Metal.beginBatch(metal.context);
            pass(new Dispatcher() {
                @Override
                public void direct(int kernel, MetalBuffer wave, int threads) {
                    Metal.addDispatch(batch, pipelines[kernel], buffers(pipelines[kernel], wave), textures, threads);
                }

                @Override
                public void indirect(int kernel, MetalBuffer wave, long argsOffset) {
                    Metal.addDispatchIndirect(batch, pipelines[kernel], buffers(pipelines[kernel], wave), textures,
                            GROUP_SIZE, work.handle, argsOffset);
                }
            });
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
            if (STATS) {
                int[] counters = new int[16];
                work.readInts(counters);
                long intersections = counters[12] & 0xFFFFFFFFL, calls = counters[13] & 0xFFFFFFFFL;
                System.out.printf(java.util.Locale.ROOT, "METAL_HWRT_STATS worldIntersections=%d hardwareCalls=%d callsPerIntersection=%.3f%n",
                        intersections, calls, intersections == 0 ? 0.0 : (double) calls / intersections);
                long queries = counters[14] & 0xFFFFFFFFL, functionCalls = counters[15] & 0xFFFFFFFFL;
                System.out.printf(java.util.Locale.ROOT, "METAL_HWRT_STATS queries=%d intersectionFunctionCalls=%d callsPerQuery=%.2f%n",
                        queries, functionCalls, queries == 0 ? 0.0 : (double) functionCalls / queries);
            }
            state.close();
            work.close();
            for (MetalBuffer[][] wave : waves) {
                for (MetalBuffer[] bounce : wave) {
                    for (MetalBuffer b : bounce) b.close();
                }
            }
            tracing.close();
            if (PROFILE && passes > 0 && !BREAKDOWN) {
                System.out.printf(java.util.Locale.ROOT, "METAL_PROFILE wavefront passes=%d gpuMsPerPass=%.2f%n", passes, gpuNanos / 1e6 / passes);
            }
            if (BREAKDOWN && passes > 0) {
                StringBuilder line = new StringBuilder("METAL_BREAKDOWN ms/pass:");
                for (int k = 0; k < kernelNanos.length; k++) {
                    line.append(String.format(java.util.Locale.ROOT, " %s=%.2f", MetalContext.WAVEFRONT_FUNCTIONS[k], kernelNanos[k] / 1e6 / passes));
                }
                System.out.println(line);
            }
        }
    }
}
