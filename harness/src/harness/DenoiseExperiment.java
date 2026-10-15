package harness;

import dev.thatredox.chunkynative.common.scene.AtmosphereData;
import dev.thatredox.chunkynative.common.scene.CameraData;
import dev.thatredox.chunkynative.common.scene.WaterData;
import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.MetalContext;
import dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.RenderContext;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

/**
 * Experiment: the MetalFX temporal denoiser on 1 spp frames of the Metal path tracer (single kernel,
 * octree, kernel define CHUNKYCL_AOV for the denoiser inputs and one subpixel offset per frame).
 * Args: scene directory, scene name, width, height, reference spp, frames, output directory,
 * jitter mode (neg: right for MetalFX; pos, flipy, flipx, zero), camera strafe per frame (0 = still camera), denoiser color input
 * (frame: each 1 spp frame; accum: the running average, still camera only), AOVs (first: first hit;
 * through: the surface behind glass), "nomotion" to leave out motion vectors, upscale factor (frames are
 * rendered at width / factor and the denoiser outputs width x height).
 * Prints PSNR against the reference (Chunky's gamma tone mapping) and writes PNGs of some frames.
 */
public class DenoiseExperiment {
  static final float NEAR = 0.1f, FAR = 4096;

  /** Kernel buffers for the render kernel (rayTracer.c), in the order it declares them. */
  static final class Gpu implements AutoCloseable {
    final MetalSceneLoader loader;
    final long context;
    final MetalBuffer projectorType, camera, seed, spp, canvas, rayDepth, water, atmosphere, res, aov;
    final int pixels;

    Gpu(MetalContext metal, Scene s) {
      loader = metal.sceneLoader;
      context = metal.context;
      pixels = s.canvasConfig.getWidth() * s.canvasConfig.getHeight();
      projectorType = MetalBuffer.ofInt(context, 0);
      camera = new MetalBuffer(context, 4L * 17);
      seed = MetalBuffer.ofInt(context, 0);
      spp = MetalBuffer.ofInt(context, 0);
      canvas = MetalBuffer.ofInts(context, CameraData.canvasConfig(s));
      rayDepth = MetalBuffer.ofInt(context, s.getRayDepth());
      water = MetalBuffer.ofFloats(context, WaterData.pack(s));
      atmosphere = MetalBuffer.ofInts(context, AtmosphereData.pack(s));
      res = new MetalBuffer(context, 4L * 3 * pixels);
      aov = new MetalBuffer(context, 4L * 16 * pixels);
    }

    /** One pass; returns the GPU time in nanoseconds. */
    long pass(long pipeline, float[] cameraSettings, int randomSeed, int bufferSpp, boolean withAov) {
      camera.writeFloats(0, cameraSettings, 0, cameraSettings.length);
      seed.writeInt(0, randomSeed);
      spp.writeInt(0, bufferSpp);
      long[] buffers = {
          projectorType.handle, camera.handle,
          loader.getOctreeDepth().handle, loader.getOctreeData().handle,
          loader.getWaterOctreeDepth().handle, loader.getWaterOctreeData().handle,
          loader.getBlockPalette().handle, loader.getQuadPalette().handle, loader.getAabbPalette().handle,
          loader.getWorldBvh().handle, loader.getActorBvh().handle, loader.getTrigPalette().handle,
          loader.getMaterialPalette().handle, loader.getSkyTexture().handle,
          loader.getSkyLight().handle, loader.getSun().handle, water.handle, atmosphere.handle,
          seed.handle, spp.handle, canvas.handle, rayDepth.handle, res.handle, aov.handle,
      };
      if (!withAov) buffers = Arrays.copyOf(buffers, buffers.length - 1);
      long[] textures = {loader.getTexturePalette().getAtlas().handle};
      return Metal.dispatch(context, pipeline, buffers, textures, pixels);
    }

    @Override
    public void close() {
      for (MetalBuffer b : new MetalBuffer[] {projectorType, camera, seed, spp, canvas, rayDepth, water, atmosphere, res, aov}) b.close();
    }
  }

  public static void main(String[] args) throws Exception {
    File dir = new File(args[0]);
    String name = args[1];
    int w = Integer.parseInt(args[2]), h = Integer.parseInt(args[3]);
    int refSpp = Integer.parseInt(args[4]), frames = Integer.parseInt(args[5]);
    File outDir = new File(args[6]);
    String jitterMode = args.length > 7 ? args[7] : "neg";
    double strafe = args.length > 8 ? Double.parseDouble(args[8]) : 0;
    String input = args.length > 9 ? args[9] : "frame";
    String aovMode = args.length > 10 ? args[10] : "first";
    boolean useMotion = !(args.length > 11 && args[11].equals("nomotion"));
    int scale = args.length > 12 ? Integer.parseInt(args[12]) : 1;
    outDir.mkdirs();

    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);
    RenderContext ctx = new RenderContext(chunky);
    ctx.setSceneDirectory(dir);
    Scene s = new Scene();
    s.loadScene(ctx, name, TaskTracker.NONE);
    s.setCanvasSize(w, h);
    float exposure = (float) s.getExposure();
    int outW = w, outH = h;

    MetalContext metal = MetalContext.get();
    metal.sceneLoader.ensureLoad(s);
    String[] defines = AtmosphereData.kernelDefines(s);
    String[] aovDefines = Arrays.copyOf(defines, defines.length + 2);
    aovDefines[defines.length] = "CHUNKYCL_AOV";
    aovDefines[defines.length + 1] = aovMode.equals("through") ? "CHUNKYCL_AOV_THROUGH" : "CHUNKYCL_AOV_FIRST";
    long plain = metal.renderPipeline(defines);
    long withAov = metal.renderPipeline(aovDefines);
    if (new CameraData(s).projectorType != 0) throw new IllegalStateException("needs a pinhole camera");

    // Reference at the output size and the camera of the last frame.
    float[] reference;
    try (Gpu gpu = new Gpu(metal, s)) {
      if (strafe != 0) s.camera().strafeRight(strafe * (frames - 1));
      reference = reference(gpu, plain, Arrays.copyOf(new CameraData(s).settings, 17), refSpp,
          new File(outDir, String.format(Locale.ROOT, "ref-%s-%dx%d-%d-%s.bin", name, w, h, refSpp, strafe * (frames - 1))));
      if (strafe != 0) s.camera().strafeLeft(strafe * (frames - 1));
      writePng(reference, w, h, exposure, new File(outDir, "reference.png"));
    }
    w /= scale;
    h /= scale;
    s.setCanvasSize(w, h);
    int px = w * h;
    try (Gpu gpu = new Gpu(metal, s); MetalFxDenoiser denoiser = new MetalFxDenoiser(w, h, outW, outH)) {

      float[] color = new float[3 * px], aov = new float[16 * px], accumulated = new float[3 * px], out = new float[3 * outW * outH];
      float[] depth = new float[px], motion = new float[2 * px], normal = new float[3 * px];
      float[] diffuse = new float[3 * px], specular = new float[3 * px], roughness = new float[px];
      float[] previous = null;
      Random random = new Random(1);
      long renderNanos = 0, denoiseNanos = 0;
      System.out.printf(Locale.ROOT, "DENOISE scene=%s %dx%d -> %dx%d ref=%d spp jitter=%s strafe=%s input=%s aov=%s%n", name, w, h, outW, outH, refSpp, jitterMode, strafe, input, aovMode);
      for (int f = 0; f < frames; f++) {
        if (f > 0 && strafe != 0) s.camera().strafeRight(strafe);
        float[] cam = Arrays.copyOf(new CameraData(s).settings, 17);
        float jx = halton(f + 1, 2), jy = halton(f + 1, 3);
        cam[15] = jx;
        cam[16] = jy;
        renderNanos += gpu.pass(withAov, cam, random.nextInt(), 0, true);
        gpu.res.readFloats(color);
        gpu.aov.readFloats(aov);
        for (int i = 0; i < 3 * px; i++) accumulated[i] = (accumulated[i] * f + color[i]) / (f + 1);

        for (int p = 0; p < px; p++) {
          float z = aov[p * 16 + 3];
          float zc = Math.max(z, NEAR);
          depth[p] = z <= 0 ? 1 : FAR * (zc - NEAR) / ((FAR - NEAR) * zc);
          for (int c = 0; c < 3; c++) {
            normal[p * 3 + c] = aov[p * 16 + 4 + c];
            diffuse[p * 3 + c] = aov[p * 16 + 7 + c];
            specular[p * 3 + c] = aov[p * 16 + 10 + c];
          }
          roughness[p] = aov[p * 16 + 13];
          motion[p * 2] = 0;
          motion[p * 2 + 1] = 0;
          if (previous != null && useMotion) {
            float[] now = project(cam, aov, p, z <= 0, w, h);
            float[] before = project(previous, aov, p, z <= 0, w, h);
            if (now != null && before != null) {
              motion[p * 2] = before[0] - now[0];
              motion[p * 2 + 1] = before[1] - now[1];
            }
          }
        }

        float ox = jx - 0.5f, oy = jy - 0.5f;
        switch (jitterMode) {
          case "neg": ox = -ox; oy = -oy; break;
          case "flipy": oy = -oy; break;
          case "flipx": ox = -ox; break;
          case "zero": ox = 0; oy = 0; break;
          default: break;
        }
        float[] params = new float[35];
        params[0] = ox;
        params[1] = oy;
        params[2] = f == 0 || input.equals("accumreset") ? 1 : 0;
        System.arraycopy(worldToView(cam), 0, params, 3, 16);
        System.arraycopy(viewToClip(cam, w, h), 0, params, 19, 16);
        denoiseNanos += denoiser.denoise(input.startsWith("accum") ? accumulated : color, depth, motion, normal, diffuse, specular, roughness, params, out);
        previous = cam;

        int n = f + 1;
        boolean last = n == frames;
        if (Integer.bitCount(n) == 1 || last) {
          String tag = String.format(Locale.ROOT, "%03d", n);
          writePng(color, w, h, exposure, new File(outDir, "frame" + tag + "-noisy.png"));
          writePng(out, outW, outH, exposure, new File(outDir, "frame" + tag + "-denoised.png"));
          if (strafe == 0) {
            writePng(accumulated, w, h, exposure, new File(outDir, "frame" + tag + "-accumulated.png"));
            System.out.printf(Locale.ROOT, "frame %3d  PSNR dB: 1 spp %.2f  accumulated %d spp %.2f  denoised %.2f%n",
                n, psnr(upscale(color, w, h, scale), reference, exposure), n, psnr(upscale(accumulated, w, h, scale), reference, exposure), psnr(out, reference, exposure));
          } else if (last) {
            System.out.printf(Locale.ROOT, "frame %3d (moving)  PSNR dB: 1 spp %.2f  denoised %.2f%n",
                n, psnr(upscale(color, w, h, scale), reference, exposure), psnr(out, reference, exposure));
          }
        }
      }
      System.out.printf(Locale.ROOT, "per frame GPU ms: render (1 spp + AOV) %.2f  denoise %.2f%n",
          renderNanos / 1e6 / frames, denoiseNanos / 1e6 / frames);
    }
    System.exit(0);
  }

  /** Nearest-neighbor upscale of an rgb image by an integer factor. */
  static float[] upscale(float[] rgb, int w, int h, int scale) {
    if (scale == 1) return rgb;
    float[] out = new float[rgb.length * scale * scale];
    int ow = w * scale;
    for (int y = 0; y < h * scale; y++) {
      for (int x = 0; x < ow; x++) {
        System.arraycopy(rgb, ((y / scale) * w + x / scale) * 3, out, (y * ow + x) * 3, 3);
      }
    }
    return out;
  }

  static float[] reference(Gpu gpu, long pipeline, float[] cam, int spp, File cache) throws IOException {
    float[] ref = new float[3 * gpu.pixels];
    if (cache.exists()) {
      try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(cache)))) {
        for (int i = 0; i < ref.length; i++) ref[i] = in.readFloat();
      }
      return ref;
    }
    Random random = new Random(12345);
    long nanos = 0;
    for (int i = 0; i < spp; i++) nanos += gpu.pass(pipeline, cam, random.nextInt(), i, false);
    gpu.res.readFloats(ref);
    System.out.printf(Locale.ROOT, "reference: %d spp, %.1f ms per pass%n", spp, nanos / 1e6 / spp);
    try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(cache)))) {
      for (float v : ref) o.writeFloat(v);
    }
    return ref;
  }

  /** Pixel position of the first hit (or sky direction) of pixel p as seen from a camera, null if behind it. */
  static float[] project(float[] cam, float[] aov, int p, boolean sky, int w, int h) {
    float dx = aov[p * 16], dy = aov[p * 16 + 1], dz = aov[p * 16 + 2];
    if (!sky) {
      dx -= cam[0];
      dy -= cam[1];
      dz -= cam[2];
    }
    // Camera space: columns of the transform (rows m1..m3 at 3..11).
    float cx = cam[3] * dx + cam[6] * dy + cam[9] * dz;
    float cy = cam[4] * dx + cam[7] * dy + cam[10] * dz;
    float cz = cam[5] * dx + cam[8] * dy + cam[11] * dz;
    if (cz <= 1e-6f) return null;
    float fovTan = cam[14];
    float halfWidth = w / (2f * h);
    float x = cx / cz / fovTan, y = cy / cz / fovTan;
    return new float[] {(x + halfWidth) * h, (y + 0.5f) * h};
  }

  /** World to view (x right, y up, looking along -z), column-major. */
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

  /** Perspective projection to Metal clip space (depth 0..1), column-major. */
  static float[] viewToClip(float[] cam, int w, int h) {
    float fovTan = cam[14];
    float halfWidth = w / (2f * h);
    float[][] m = new float[4][4];
    m[0][0] = 1 / (fovTan * halfWidth);
    m[1][1] = 1 / (fovTan * 0.5f);
    m[2][2] = FAR / (NEAR - FAR);
    m[2][3] = NEAR * FAR / (NEAR - FAR);
    m[3][2] = -1;
    return columnMajor(m);
  }

  static float[] columnMajor(float[][] m) {
    float[] a = new float[16];
    for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) a[c * 4 + r] = m[r][c];
    return a;
  }

  static float halton(int index, int base) {
    float f = 1, r = 0;
    while (index > 0) {
      f /= base;
      r += f * (index % base);
      index /= base;
    }
    return r;
  }

  static float tonemap(float v, float exposure) {
    return (float) Math.pow(Math.min(1, Math.max(0, v * exposure)), 1 / 2.2);
  }

  static double psnr(float[] a, float[] ref, float exposure) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
      double d = tonemap(a[i], exposure) - tonemap(ref[i], exposure);
      sum += d * d;
    }
    return 10 * Math.log10(1 / (sum / a.length));
  }

  static void writePng(float[] rgb, int w, int h, float exposure, File file) throws IOException {
    BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int i = (y * w + x) * 3;
        int r = Math.round(tonemap(rgb[i], exposure) * 255);
        int g = Math.round(tonemap(rgb[i + 1], exposure) * 255);
        int b = Math.round(tonemap(rgb[i + 2], exposure) * 255);
        img.setRGB(x, y, (r << 16) | (g << 8) | b);
      }
    }
    ImageIO.write(img, "png", file);
  }
}
