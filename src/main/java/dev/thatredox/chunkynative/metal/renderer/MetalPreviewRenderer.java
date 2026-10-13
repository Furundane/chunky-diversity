package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.common.scene.AtmosphereData;
import dev.thatredox.chunkynative.common.scene.CameraData;
import dev.thatredox.chunkynative.common.scene.WaterData;
import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.renderer.Renderer;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.function.BooleanSupplier;

/**
 * Preview renderer on Metal. It shares the scene with the Metal path tracer, so using Metal for both
 * keeps a single copy of the scene on the GPU (with the OpenCL preview, the scene is also uploaded to
 * OpenCL).
 */
public class MetalPreviewRenderer implements Renderer {
    public static final String ID = "ChunkyMetalPreviewRenderer";

    private BooleanSupplier postRender = () -> true;

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getName() {
        return "ChunkyCL (Metal) preview";
    }

    @Override
    public String getDescription() {
        return "GPU preview using Metal";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        MetalContext metal = MetalContext.get();
        MetalSceneLoader sceneLoader = metal.sceneLoader;
        Scene scene = manager.bufferedScene;
        int[] imageData = scene.getBackBuffer().data;
        long context = metal.context;

        sceneLoader.ensureLoad(scene);

        CameraData camera = new CameraData(scene);
        float[] cameraSettings = camera.needsRayGeneration() ? CameraData.generateRays(scene, false) : camera.settings;
        try (MetalBuffer projectorType = MetalBuffer.ofInt(context, camera.projectorType);
             MetalBuffer cameraBuffer = MetalBuffer.ofFloats(context, cameraSettings);
             MetalBuffer water = MetalBuffer.ofFloats(context, WaterData.pack(scene));
             MetalBuffer atmosphere = MetalBuffer.ofInts(context, AtmosphereData.pack(scene));
             MetalBuffer canvasConfig = MetalBuffer.ofInts(context, CameraData.canvasConfig(scene));
             MetalBuffer result = new MetalBuffer(context, 4L * imageData.length);
             BlockTracing tracing = new BlockTracing(sceneLoader)) {
            long pipeline = metal.previewPipeline(tracing.defines(AtmosphereData.kernelDefines(scene)));
            tracing.bind(context, pipeline, sceneLoader);

            // Buffers and textures in the order the preview kernel declares them (rayTracer.c).
            long[] buffers = tracing.buffers(new long[] {
                    projectorType.handle,
                    cameraBuffer.handle,

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

                    canvasConfig.handle,
                    result.handle,
            });
            long[] textures = {
                    sceneLoader.getTexturePalette().getAtlas().handle,
            };
            Metal.dispatch(context, pipeline, buffers, textures, imageData.length);
            result.readInts(imageData);
        }

        manager.redrawScreen();
        postRender.getAsBoolean();
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        MetalContext.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
    }
}
