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
import se.llbit.log.Log;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BooleanSupplier;

/**
 * Path traced preview on Metal, denoised with the MetalFX temporal denoised scaler (macOS 26). Each
 * frame is one sample per pixel of the render kernel. While the camera moves, frames are rendered at
 * a lower resolution and upscaled; when it stops, a few more upscaled frames, then frames at full
 * resolution until the image settles. The denoiser keeps a history across frames and follows camera
 * moves with motion vectors. Falls back to the Metal preview where MetalFX can't be used or the
 * projection isn't pinhole.
 */
public class MetalDenoisedPreviewRenderer implements Renderer {
    public static final String ID = "ChunkyMetalDenoisedPreviewRenderer";

    /** Kernel define for the denoiser inputs (rayTracer.c). */
    private static final String AOV_DEFINE = "CHUNKYCL_AOV";
    /** Resolution divisor while the camera moves (1 to not upscale). */
    private static final int UPSCALE = Math.max(1, Math.min(3, Integer.getInteger("chunkymetal.previewUpscale", 2)));
    /** Upscaled frames after a change, then full resolution frames until the preview stops. */
    private static final int UPSCALED_FRAMES = 4;
    private static final int FULL_FRAMES = Math.max(1, Integer.getInteger("chunkymetal.previewFrames", 16));
    // -Dchunkycl.profilePasses=true prints the GPU time per frame when the preview stops.
    private static final boolean PROFILE = Boolean.getBoolean("chunkycl.profilePasses");
    // Diagnostics: -Dchunkymetal.previewJitter=neg|pos|flipx|flipy|off (offset passed to MetalFX; off: no jitter),
    // -Dchunkymetal.previewDump=<directory> writes every frame as a PNG (with -Dchunkymetal.previewDumpShown=true
    // the buffer shown on the canvas after the frame, instead of the frame rendered).
    private static final String JITTER = System.getProperty("chunkymetal.previewJitter", "neg");
    private static final String DUMP = System.getProperty("chunkymetal.previewDump", "");
    private static final boolean DUMP_SHOWN = Boolean.getBoolean("chunkymetal.previewDumpShown");
    private int dumped = 0;
    /** Depth range for the denoiser's depth input. */
    private static final float NEAR = 0.1f, FAR = 4096;

    private final MetalPreviewRenderer fallback = new MetalPreviewRenderer();
    private BooleanSupplier postRender = () -> true;
    private final Stage[] stages = new Stage[4];
    private final Random random = new Random();
    private boolean unsupported = false;
    private boolean resetHistory = true;
    /** Camera of the last rendered frame (without jitter). */
    private float[] renderedCamera;

    @Override
    public String getId() {
        return ID;
    }

    @Override
    public String getName() {
        return "ChunkyCL (Metal) path traced preview";
    }

    @Override
    public String getDescription() {
        return "Path traced GPU preview, denoised and upscaled with MetalFX (macOS 26)";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        MetalContext.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
        if (reason == ResetReason.SCENE_LOADED || reason == ResetReason.MATERIALS_CHANGED) {
            resetHistory = true;
        } else if (reason == ResetReason.SETTINGS_CHANGED || reason == ResetReason.SETTINGS_CHANGED_SOFT) {
            // The history follows camera moves; other changes start over.
            CameraData camera = new CameraData(manager.bufferedScene);
            if (renderedCamera == null || Arrays.equals(camera.settings, renderedCamera)) resetHistory = true;
        }
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        MetalContext metal = MetalContext.get();
        Scene scene = manager.bufferedScene;
        metal.sceneLoader.ensureLoad(scene);
        CameraData camera = new CameraData(scene);
        if (unsupported || camera.needsRayGeneration()) {
            renderFallback(manager);
            return;
        }
        if (resetHistory) {
            for (Stage stage : stages) if (stage != null) stage.previousCamera = null;
            resetHistory = false;
        }
        renderedCamera = camera.settings;
        long context = metal.context;

        try (BlockTracing tracing = new BlockTracing(metal.sceneLoader);
             MetalBuffer water = MetalBuffer.ofFloats(context, WaterData.pack(scene));
             MetalBuffer atmosphere = MetalBuffer.ofInts(context, AtmosphereData.pack(scene));
             MetalBuffer rayDepth = MetalBuffer.ofInt(context, scene.getRayDepth())) {
            String[] base = tracing.defines(AtmosphereData.kernelDefines(scene));
            String[] defines = Arrays.copyOf(base, base.length + 1);
            defines[base.length] = AOV_DEFINE;
            long pipeline = metal.renderPipeline(defines);
            tracing.bind(context, pipeline);

            int upscaled = UPSCALE > 1 ? UPSCALED_FRAMES : 0;
            long[] nanos = new long[4];
            int[] frames = new int[4];
            for (int frame = 0; frame < upscaled + FULL_FRAMES; frame++) {
                Stage stage = stage(metal, frame < upscaled ? UPSCALE : 1, scene);
                if (stage == null) {
                    renderFallback(manager);
                    return;
                }
                long[] buffers = tracing.buffers(pipeline, new long[] {
                        stage.projectorType.handle,
                        stage.camera.handle,

                        metal.sceneLoader.getOctreeDepth().handle,
                        metal.sceneLoader.getOctreeData().handle,
                }, new long[] {
                        metal.sceneLoader.getWaterOctreeDepth().handle,
                        metal.sceneLoader.getWaterOctreeData().handle,

                        metal.sceneLoader.getBlockPalette().handle,
                        metal.sceneLoader.getQuadPalette().handle,
                        metal.sceneLoader.getAabbPalette().handle,

                        metal.sceneLoader.getWorldBvh().handle,
                        metal.sceneLoader.getActorBvh().handle,
                        metal.sceneLoader.getTrigPalette().handle,

                        metal.sceneLoader.getMaterialPalette().handle,
                        metal.sceneLoader.getSkyTexture().handle,

                        metal.sceneLoader.getSkyLight().handle,
                        metal.sceneLoader.getSun().handle,
                        water.handle,
                        atmosphere.handle,

                        stage.seed.handle,
                        stage.spp.handle,
                        stage.canvasConfig.handle,
                        rayDepth.handle,
                        stage.color.handle,
                        stage.aov.handle,
                });
                long[] times = stage.render(metal, pipeline, buffers, camera.settings, scene, random.nextInt());
                nanos[stage.scale * 2 - 2] += times[0];
                nanos[stage.scale * 2 - 1] += times[1];
                frames[stage.scale - 1]++;
                // redrawScreen swaps the front and back buffers, so take the back buffer anew for every frame
                // (writing to the one fetched before the loop would alternate the canvas between this frame
                // and an old one).
                int[] imageData = scene.getBackBuffer().data;
                stage.argb.readInts(imageData);
                if (!DUMP.isEmpty() && !DUMP_SHOWN) dumpFrame(imageData, stage);
                drawCrosshair(scene, imageData);
                manager.redrawScreen();
                if (!DUMP.isEmpty() && DUMP_SHOWN) manager.withBufferedImage(bitmap -> dumpFrame(bitmap.data, stage));
                boolean changed = postRender.getAsBoolean();
                if (changed || frame == upscaled + FULL_FRAMES - 1) {
                    if (PROFILE) printProfile(nanos, frames);
                    return;
                }
            }
        }
    }

    private void dumpFrame(int[] imageData, Stage stage) {
        try {
            java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(stage.outWidth, stage.outHeight,
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            image.setRGB(0, 0, stage.outWidth, stage.outHeight, imageData, 0, stage.outWidth);
            java.io.File dir = new java.io.File(DUMP);
            dir.mkdirs();
            javax.imageio.ImageIO.write(image, "png", new java.io.File(dir, String.format("%04d-s%d.png", dumped++, stage.scale)));
        } catch (java.io.IOException e) {
            Log.warn("Could not write a preview frame", e);
        }
    }

    private static void printProfile(long[] nanos, int[] frames) {
        StringBuilder line = new StringBuilder("METAL_PREVIEW");
        for (int scale = 1; scale <= 2; scale++) {
            int n = frames[scale - 1];
            if (n == 0) continue;
            line.append(String.format(java.util.Locale.ROOT, " 1/%d: frames=%d renderMs=%.2f denoiseMs=%.2f", scale, n,
                    nanos[scale * 2 - 2] / 1e6 / n, nanos[scale * 2 - 1] / 1e6 / n));
        }
        System.out.println(line);
    }

    private void renderFallback(DefaultRenderManager manager) throws InterruptedException {
        fallback.setPostRender(postRender);
        fallback.render(manager);
    }

    /** The resources for frames at 1 / scale of the canvas size, created when needed. */
    private Stage stage(MetalContext metal, int scale, Scene scene) {
        int width = scene.canvasConfig.getWidth();
        int height = scene.canvasConfig.getHeight();
        Stage stage = stages[scale];
        if (stage != null && (stage.outWidth != width || stage.outHeight != height)) {
            stage.close();
            stage = stages[scale] = null;
        }
        if (stage == null) {
            int inWidth = Math.max(1, width / scale), inHeight = Math.max(1, height / scale);
            long denoiser = Metal.newDenoiser(metal.context, inWidth, inHeight, width, height);
            if (denoiser == 0) {
                Log.info("ChunkyCL Metal: MetalFX denoising is not available here (needs macOS 26), using the Metal preview");
                unsupported = true;
                return null;
            }
            stage = stages[scale] = new Stage(metal.context, denoiser, scale, scene, inWidth, inHeight);
        }
        return stage;
    }

    private static void drawCrosshair(Scene scene, int[] imageData) {
        int width = scene.canvasConfig.getWidth();
        int height = scene.canvasConfig.getHeight();
        int fullWidth = scene.canvasConfig.getCropWidth();
        int fullHeight = scene.canvasConfig.getCropHeight();
        int cx = fullWidth / 2 - scene.canvasConfig.getCropX();
        int cy = fullHeight / 2 - scene.canvasConfig.getCropY();
        for (int d = -5; d <= 5; d++) {
            if (cx >= 0 && cx < width && cy + d >= 0 && cy + d < height) imageData[(cy + d) * width + cx] = 0xFFFFFFFF;
            if (cy >= 0 && cy < height && cx + d >= 0 && cx + d < width) imageData[cy * width + cx + d] = 0xFFFFFFFF;
        }
    }

    /** Tone mapping in the unpack kernel for the scene's post-processing filter (metal_bridge.m). */
    private static int toneMapping(Scene scene) {
        switch (scene.getPostProcessingFilter().getId()) {
            case "TONEMAP1": return 1;
            case "TONEMAP2": return 2;
            case "TONEMAP3": return 3;
            case "NONE": return 4;
            default: return 0;  // Gamma correction, also used for the others
        }
    }

    /** Buffers and the denoiser for one input size. */
    private static final class Stage implements AutoCloseable {
        final long denoiser;
        final int scale, inWidth, inHeight, outWidth, outHeight;
        final MetalBuffer projectorType, camera, seed, spp, canvasConfig, color, aov, argb;
        final int[] fullConfig;
        /** Camera of the previous frame of this stage, null to start a new history. */
        float[] previousCamera;
        int frameIndex = 0;

        Stage(long context, long denoiser, int scale, Scene scene, int inWidth, int inHeight) {
            this.denoiser = denoiser;
            this.scale = scale;
            this.inWidth = inWidth;
            this.inHeight = inHeight;
            outWidth = scene.canvasConfig.getWidth();
            outHeight = scene.canvasConfig.getHeight();
            fullConfig = new int[] {
                    inWidth, inHeight,
                    Math.max(1, scene.canvasConfig.getCropWidth() / scale), Math.max(1, scene.canvasConfig.getCropHeight() / scale),
                    scene.canvasConfig.getCropX() / scale, scene.canvasConfig.getCropY() / scale,
            };
            projectorType = MetalBuffer.ofInt(context, 0);
            camera = new MetalBuffer(context, 4L * 17);
            seed = MetalBuffer.ofInt(context, 0);
            spp = MetalBuffer.ofInt(context, 0);
            canvasConfig = MetalBuffer.ofInts(context, fullConfig);
            color = new MetalBuffer(context, 4L * 3 * inWidth * inHeight);
            aov = new MetalBuffer(context, 4L * 16 * inWidth * inHeight);
            argb = new MetalBuffer(context, 4L * outWidth * outHeight);
        }

        /** Renders and denoises one frame into argb; returns the GPU times of both in nanoseconds. */
        long[] render(MetalContext metal, long pipeline, long[] buffers, float[] cameraSettings, Scene scene, int randomSeed) {
            // One subpixel offset for the whole frame (Halton 2, 3), read by the kernel after the camera settings.
            frameIndex = frameIndex % 64 + 1;
            float jx = halton(frameIndex, 2), jy = halton(frameIndex, 3);
            if (JITTER.equals("off")) jx = jy = 0.5f;
            float[] settings = Arrays.copyOf(cameraSettings, 17);
            settings[15] = jx;
            settings[16] = jy;
            camera.writeFloats(0, settings, 0, settings.length);
            seed.writeInt(0, randomSeed);
            // Buffer spp 0: the color buffer holds this frame only.
            long renderNanos = Metal.dispatch(metal.context, pipeline, buffers,
                    new long[] {metal.sceneLoader.getTexturePalette().getAtlas().handle}, inWidth * inHeight);

            float[] params = new float[43];
            System.arraycopy(cameraSettings, 0, params, 0, 15);
            if (previousCamera != null) System.arraycopy(previousCamera, 0, params, 15, 15);
            for (int i = 0; i < 6; i++) params[30 + i] = fullConfig[i];
            params[36] = NEAR;
            params[37] = FAR;
            params[38] = previousCamera != null ? 1 : 0;
            params[39] = (float) scene.getExposure();
            params[40] = toneMapping(scene);
            params[41] = outWidth;
            params[42] = outHeight;

            float[] scalerParams = new float[35];
            // MetalFX takes the offset back to the pixel center.
            scalerParams[0] = 0.5f - jx;
            scalerParams[1] = 0.5f - jy;
            if (JITTER.equals("pos") || JITTER.equals("flipx")) scalerParams[0] = -scalerParams[0];
            if (JITTER.equals("pos") || JITTER.equals("flipy")) scalerParams[1] = -scalerParams[1];
            scalerParams[2] = previousCamera == null ? 1 : 0;
            System.arraycopy(worldToView(cameraSettings), 0, scalerParams, 3, 16);
            System.arraycopy(viewToClip(cameraSettings, fullConfig[2], fullConfig[3]), 0, scalerParams, 19, 16);
            long denoiseNanos = Metal.denoise(metal.context, denoiser, color.handle, aov.handle, argb.handle, params, scalerParams);
            previousCamera = cameraSettings;
            return new long[] {renderNanos, denoiseNanos};
        }

        @Override
        public void close() {
            for (MetalBuffer b : new MetalBuffer[] {projectorType, camera, seed, spp, canvasConfig, color, aov, argb}) b.close();
            Metal.release(denoiser);
        }
    }

    /** World to view (x right, y up, looking along -z), column-major, from the camera settings. */
    static float[] worldToView(float[] cam) {
        float[][] v = new float[4][4];
        float[] flip = {1, -1, -1};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) v[r][c] = flip[r] * cam[3 + c * 3 + r];
            v[r][3] = -(v[r][0] * cam[0] + v[r][1] * cam[1] + v[r][2] * cam[2]);
        }
        v[3][3] = 1;
        return columnMajor(v);
    }

    /** Perspective projection to Metal clip space (depth 0 to 1), column-major. */
    static float[] viewToClip(float[] cam, int fullWidth, int fullHeight) {
        float fovTan = cam[14];
        float halfWidth = fullWidth / (2f * fullHeight);
        float[][] m = new float[4][4];
        m[0][0] = 1 / (fovTan * halfWidth);
        m[1][1] = 1 / (fovTan * 0.5f);
        m[2][2] = FAR / (NEAR - FAR);
        m[2][3] = NEAR * FAR / (NEAR - FAR);
        m[3][2] = -1;
        return columnMajor(m);
    }

    private static float[] columnMajor(float[][] m) {
        float[] a = new float[16];
        for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) a[c * 4 + r] = m[r][c];
        return a;
    }

    private static float halton(int index, int base) {
        float f = 1, r = 0;
        while (index > 0) {
            f /= base;
            r += f * (index % base);
            index /= base;
        }
        return r;
    }
}
