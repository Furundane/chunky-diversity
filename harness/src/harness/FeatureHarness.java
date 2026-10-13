package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.block.minecraft.Water;
import se.llbit.chunky.chunk.BlockPalette;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.projection.ProjectionMode;
import se.llbit.chunky.renderer.scene.FogMode;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;
import se.llbit.math.Octree;
import se.llbit.math.Vector3;
import se.llbit.nbt.CompoundTag;
import se.llbit.nbt.StringTag;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Renders small synthetic scenes with Chunky's CPU path tracer and the ChunkyCL OpenCL renderer
 * and reports which scene features the GPU renderer ignores or renders differently.
 */
public class FeatureHarness {
  static final int W = 320, H = 180;
  static final int CPU_SPP = 64, GPU_SPP = 256;
  static final String CPU = DefaultRenderManager.ChunkyPathTracerID;
  // -Pgpu=ChunkyMetalRenderer and -PharnessOut=out-metal (see harness-init.gradle) test another GPU backend.
  static final String GPU = System.getProperty("harness.gpu", "ChunkyClRenderer");
  static final File OUT = new File(System.getProperty("harness.out", "out"));
  static final List<String> logLines = Collections.synchronizedList(new ArrayList<>());
  static final StringBuilder report = new StringBuilder();
  static Chunky chunky;
  static long startTime;
  static Set<String> selected = new HashSet<>();

  static boolean skip(String name) {
    return !selected.isEmpty() && !selected.contains(name);
  }

  public static void main(String[] args) throws Exception {
    startTime = System.currentTimeMillis();
    OUT.mkdirs();
    selected.addAll(Arrays.asList(args));
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) {
        if (level == Level.ERROR || level == Level.WARNING) logLines.add(level + ": " + message);
      }
    }, Level.ERROR, Level.WARNING, Level.INFO);

    Chunky.loadDefaultTextures();
    ChunkyOptions options = ChunkyOptions.getDefaults();
    options.renderThreads = Runtime.getRuntime().availableProcessors();
    chunky = new Chunky(options);
    new ChunkyCl().attach(chunky);

    report.append("kind\tcase\tresult\tdetail\n");

    // --- Cross-renderer comparisons: same scene, CPU vs GPU ---
    compare("baseline_day", s -> {});
    compare("hdr_solid_sky_2.5", s -> {
      s.sky().setSkyMode(Sky.SkyMode.SOLID_COLOR);
      s.sky().setColor(new Vector3(2.5, 2.5, 2.5));
    });
    compare("solid_sky_0.5", s -> {
      s.sky().setSkyMode(Sky.SkyMode.SOLID_COLOR);
      s.sky().setColor(new Vector3(0.5, 0.5, 0.5));
    });
    compare("night_emitters", FeatureHarness::night);
    for (ProjectionMode p : new ProjectionMode[] {ProjectionMode.FISHEYE, ProjectionMode.PANORAMIC, ProjectionMode.PARALLEL}) {
      compare("projection_" + p.name(), s -> s.camera().setProjectionMode(p));
    }
    compare("dof_pinhole", s -> { s.camera().setDof(1.5); s.camera().setSubjectDistance(30); });
    compare("underwater_camera", s -> {
      s.camera().setPosition(new Vector3(32, 3.5, 14));
      s.camera().setView(Math.PI / 2, -Math.PI / 2 + 0.15, 0);
    });
    compare("underwater_camera_no_sun_sampling", s -> {
      s.setSunSamplingStrategy(SunSamplingStrategy.OFF);
      s.camera().setPosition(new Vector3(32, 3.5, 14));
      s.camera().setView(Math.PI / 2, -Math.PI / 2 + 0.15, 0);
    });
    compare("baseline_no_sun_sampling", s -> s.setSunSamplingStrategy(SunSamplingStrategy.OFF));
    compare("water_wave_shading_default", s -> s.setWaterShadingStrategy(WaterShadingStrategy.SIMPLEX));
    compare("water_waves_simplex_close", s -> { waterView(s); s.setWaterShadingStrategy(WaterShadingStrategy.SIMPLEX); });
    compare("water_waves_tiled_close", s -> { waterView(s); s.setWaterShadingStrategy(WaterShadingStrategy.TILED_NORMALMAP); });
    compare("cyan_stained_glass_wall", s -> wall(s, tag("minecraft:cyan_stained_glass"), 2));
    compare("cyan_stained_glass_pane_wall", s -> wall(s, tag("minecraft:cyan_stained_glass_pane",
        "east", "true", "west", "true", "north", "false", "south", "false", "waterlogged", "false"), 1));
    compare("clear_glass_wall", s -> wall(s, tag("minecraft:glass"), 1));
    compare("fog_uniform", s -> { s.setFogMode(FogMode.UNIFORM); s.setFogDensity(1); });
    compare("fog_layered", FeatureHarness::layeredFog);
    compare("clouds", FeatureHarness::clouds);
    compare("clouds_overhead", s -> { clouds(s); lookUp(s); });
    compare("dof_fisheye", s -> {
      s.camera().setProjectionMode(ProjectionMode.FISHEYE);
      s.camera().setDof(1.5); s.camera().setSubjectDistance(30);
    });

    // --- Toggle tests: does changing a setting change the GPU image at all? ---
    toggle("water_blocks", s -> {}, s -> {}, false, true);
    toggle("fog_uniform", s -> {}, s -> { s.setFogMode(FogMode.UNIFORM); s.setFogDensity(0.05); });
    toggle("fog_layered", s -> {}, FeatureHarness::layeredFog);
    toggle("clouds_enabled", s -> { s.sky().setCloudYOffset(30); lookUp(s); }, s -> { clouds(s); lookUp(s); });
    toggle("sky_light_modifier", s -> {}, s -> s.sky().setSkyLight(0.25));
    toggle("sun_intensity", s -> {}, s -> s.sun().setIntensity(3.0));
    toggle("sun_sampling_off_vs_fast", s -> s.setSunSamplingStrategy(SunSamplingStrategy.OFF),
        s -> s.setSunSamplingStrategy(SunSamplingStrategy.FAST));
    toggle("emitter_intensity", s -> { night(s); s.setEmitterIntensity(1); },
        s -> { night(s); s.setEmitterIntensity(10); });
    // At night with emitters disabled there is no light at all, so a black image is the expected result.
    allowBlack = true;
    toggle("emitters_enabled", s -> { night(s); s.setEmittersEnabled(false); },
        s -> { night(s); s.setEmittersEnabled(true); });
    allowBlack = false;
    toggle("water_waves", FeatureHarness::waterView,
        s -> { waterView(s); s.setWaterShadingStrategy(WaterShadingStrategy.SIMPLEX); });
    toggle("water_waves_tiled", FeatureHarness::waterView,
        s -> { waterView(s); s.setWaterShadingStrategy(WaterShadingStrategy.TILED_NORMALMAP); });
    toggle("water_visibility", s -> s.setWaterVisibility(9), s -> s.setWaterVisibility(1), true, true);
    toggle("water_opacity", s -> s.setWaterOpacity(0.1), s -> s.setWaterOpacity(0.8));
    // With sun sampling OFF, the CPU reference at CPU_SPP is so noisy (sun texture fireflies) that two renders
    // of the same scene differ as much as this toggle changes the image, so compare each setting instead.
    for (double opacity : new double[] {0.1, 0.8}) {
      compare("water_opacity_" + opacity + "_no_sun_sampling",
          s -> { s.setSunSamplingStrategy(SunSamplingStrategy.OFF); s.setWaterOpacity(opacity); });
    }
    toggle("custom_water_color", s -> {}, s -> {
      s.setUseCustomWaterColor(true);
      s.setWaterColor(new Vector3(0.8, 0.1, 0.1));
    });
    toggle("ray_depth_control", s -> s.setRayDepth(1), s -> s.setRayDepth(8));
    toggle("branch_count", s -> s.setBranchCount(1), s -> s.setBranchCount(4));

    // --- Robustness: non-packed octree ---
    robustness("octree_NODE", "NODE");
    robustness("octree_BIGPACKED", "BIGPACKED");
    finalize = false;
    robustness("headless_finalize_off", "PACKED");
    finalize = true;

    try (PrintWriter w = new PrintWriter(new File(OUT, selected.isEmpty() ? "report.tsv" : "report-partial.tsv"), "UTF-8")) {
      w.print(report);
    }
    try (PrintWriter w = new PrintWriter(new File(OUT, "log.txt"), "UTF-8")) {
      logLines.forEach(w::println);
    }
    System.out.println(report);
    System.out.printf("CPU reference renders from cache: %d, total time %.1fs%n", cacheHits,
        (System.currentTimeMillis() - startTime) / 1000.0);
    if (GPU.contains("Metal")) {
      System.out.println("Metal block tracing: " + dev.thatredox.chunkynative.metal.BlockAcceleration.current()
          + ", " + dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader.accelerationStatus());
    }
    System.exit(0);
  }

  // ---------------------------------------------------------------- scene

  static void night(Scene s) {
    s.sky().setSkyMode(Sky.SkyMode.SOLID_COLOR);
    s.sky().setColor(new Vector3(0, 0, 0));
    s.sun().setIntensity(0);
    s.sun().setDrawTexture(false);
    // Not OFF: with OFF, Chunky still lights diffuse surfaces with the sun texture at intensity 0
    // (Sun.intersectDiffuse ignores the intensity), which would hide the emitters. The sun sampling
    // modes have their own cases.
    s.setSunSamplingStrategy(SunSamplingStrategy.FAST);
    s.setEmittersEnabled(true);
    s.setEmitterSamplingStrategy(EmitterSamplingStrategy.NONE);
  }

  /** One fog layer around the height of the feature blocks. */
  static void layeredFog(Scene s) {
    s.setFogMode(FogMode.LAYERED);
    s.fog.addLayer();
    s.fog.setY(0, 6);
    s.fog.setBreadth(0, 3);
    s.fog.setDensity(0, 1);
  }

  /** Clouds low enough to be in view, shifted so that some are right above the scene. */
  static void clouds(Scene s) {
    s.sky().setCloudsEnabled(true);
    s.sky().setCloudYOffset(30);
    s.sky().setCloudXOffset(90);
  }

  /** Camera just above the pool, looking over the water at a shallow angle (mostly reflections). */
  static void waterView(Scene s) {
    s.camera().setPosition(new Vector3(32, 6, 13));
    s.camera().setView(Math.PI / 2, -Math.PI / 2 + 0.3, 0);
  }

  /** Camera looking up into the sky. */
  static void lookUp(Scene s) {
    s.camera().setView(Math.PI / 2, -Math.PI / 2 - 0.7, 0);
  }

  static CompoundTag tag(String name, String... properties) {
    CompoundTag t = new CompoundTag();
    t.add("Name", new StringTag(name));
    if (properties.length > 0) {
      CompoundTag props = new CompoundTag();
      for (int i = 0; i < properties.length; i += 2) props.add(properties[i], new StringTag(properties[i + 1]));
      t.add("Properties", props);
    }
    return t;
  }

  /** A wall of the given block between the camera and the row of feature blocks. */
  static void wall(Scene s, CompoundTag block, int thickness) {
    int id = s.getPalette().put(block);
    for (int x = 12; x < 52; x++)
      for (int y = 5; y < 14; y++)
        for (int z = 34; z < 34 + thickness; z++) s.getWorldOctree().set(id, x, y, z);
  }

  public static Scene buildScene(String octreeImpl, boolean water) throws Exception {
    Scene s = new Scene();
    s.setOctreeImplementation(octreeImpl);
    s.setCanvasSize(W, H);
    s.setName("harness");
    s.setBiomeColorsEnabled(false);
    // ChunkyCL has no wave shading; compare against still water unless a case says otherwise.
    s.setWaterShadingStrategy(WaterShadingStrategy.STILL);
    Octree world = new Octree(octreeImpl, 6);
    Octree waterTree = new Octree(octreeImpl, 6);
    setField(s, "worldOctree", world);
    setField(s, "waterOctree", waterTree);

    BlockPalette p = s.getPalette();
    int stone = p.put(tag("minecraft:stone"));
    int planks = p.put(tag("minecraft:oak_planks"));
    int glass = p.put(tag("minecraft:glass"));
    int redGlass = p.put(tag("minecraft:red_stained_glass"));
    int glow = p.put(tag("minecraft:glowstone"));
    int gold = p.put(tag("minecraft:gold_block"));
    int lava = p.getLavaId(0, 1 << Water.FULL_BLOCK);
    int fullWater = p.getWaterId(0, 1 << Water.FULL_BLOCK);
    int surfaceWater = p.getWaterId(0, 0); // still water surface: all corners at 14/16

    for (int x = 0; x < 64; x++)
      for (int z = 0; z < 64; z++)
        for (int y = 0; y < 5; y++) world.set(stone, x, y, z);
    // water pool in front of the camera
    for (int x = 16; x < 48; x++)
      for (int z = 12; z < 30; z++)
        for (int y = 2; y < 5; y++) {
          world.set(p.airId, x, y, z);
          if (water) waterTree.set(y == 4 ? surfaceWater : fullWater, x, y, z);
        }
    // a row of feature blocks
    int[] row = {glass, redGlass, glow, lava, gold, planks};
    for (int i = 0; i < row.length; i++)
      for (int y = 5; y < 8; y++) {
        world.set(row[i], 18 + i * 5, y, 38);
        world.set(row[i], 19 + i * 5, y, 38);
      }
    // back wall
    for (int x = 8; x < 56; x++)
      for (int y = 5; y < 18; y++) world.set(planks, x, y, 48);

    s.camera().setPosition(new Vector3(32, 13, 2));
    s.camera().setView(Math.PI / 2, -Math.PI / 2 + 0.38, 0);
    s.setRenderMode(RenderMode.RENDERING);
    return s;
  }

  static void setField(Object o, String name, Object value) throws Exception {
    Field f = Scene.class.getDeclaredField(name);
    f.setAccessible(true);
    f.set(o, value);
  }

  // ---------------------------------------------------------------- render

  static class Result {
    double[] samples;
    String error;
    long millis;
  }

  static boolean finalize = true;
  static boolean allowBlack = false;

  // CPU reference renders don't change while the GPU plugin is being worked on, so they are cached on disk.
  // The key includes the harness source and the Chunky core jar, so changing either re-renders them.
  static final File CPU_CACHE = new File("cache/cpu");
  static String cacheSalt;
  static int cacheHits = 0;

  static String cacheSalt() throws Exception {
    if (cacheSalt == null) {
      java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-1");
      md.update(java.nio.file.Files.readAllBytes(new File("src/harness/FeatureHarness.java").toPath()));
      File core = new File(Scene.class.getProtectionDomain().getCodeSource().getLocation().toURI());
      if (core.isFile()) md.update(java.nio.file.Files.readAllBytes(core.toPath()));
      md.update(String.format("%dx%d@%d", W, H, CPU_SPP).getBytes());
      StringBuilder hex = new StringBuilder();
      for (byte b : md.digest()) hex.append(String.format("%02x", b));
      cacheSalt = hex.substring(0, 12);
    }
    return cacheSalt;
  }

  static Result renderCpu(String key, Consumer<Scene> mod, boolean water, String octreeImpl) {
    if (Boolean.getBoolean("harness.noCache")) return render(CPU, mod, water, octreeImpl);
    File file;
    try {
      file = new File(CPU_CACHE, key + "-" + (finalize ? "f" : "h") + "-" + cacheSalt() + ".bin");
    } catch (Exception e) {
      return render(CPU, mod, water, octreeImpl);
    }
    if (file.isFile()) {
      try (java.io.DataInputStream in = new java.io.DataInputStream(
          new java.io.BufferedInputStream(new java.io.FileInputStream(file)))) {
        Result r = new Result();
        r.samples = new double[in.readInt()];
        for (int i = 0; i < r.samples.length; i++) r.samples[i] = in.readDouble();
        cacheHits++;
        return r;
      } catch (Exception e) {
        // Fall through and re-render
      }
    }
    Result r = render(CPU, mod, water, octreeImpl);
    if (r.error == null && r.samples != null) {
      CPU_CACHE.mkdirs();
      try (java.io.DataOutputStream out = new java.io.DataOutputStream(
          new java.io.BufferedOutputStream(new java.io.FileOutputStream(file)))) {
        out.writeInt(r.samples.length);
        for (double v : r.samples) out.writeDouble(v);
      } catch (Exception e) {
        file.delete();
      }
    }
    return r;
  }

  static Result render(String renderer, Consumer<Scene> mod, boolean water, String octreeImpl) {
    Result r = new Result();
    int logStart = logLines.size();
    long t0 = System.currentTimeMillis();
    DefaultRenderManager rm = null;
    try {
      Scene s = buildScene(octreeImpl, water);
      mod.accept(s);
      s.setRenderer(renderer);
      s.setTargetSpp(renderer.equals(GPU) ? GPU_SPP : CPU_SPP);
      s.setRenderMode(RenderMode.RENDERING);
      // Mimic the GUI (visible canvas). Headless mode leaves this off, which makes ChunkyCL
      // drop every GPU pass since its last merge (see the headless_finalize_off case).
      s.setBufferFinalization(finalize);

      RenderContext ctx = new RenderContext(chunky);
      rm = new DefaultRenderManager(ctx, true);
      rm.setSnapshotControl(new SnapshotControl() {
        // Only drives buffer finalization in DefaultRenderManager; no file is written.
        // ChunkyCL relies on this event to merge its last GPU pass into the sample buffer.
        @Override public boolean saveSnapshot(Scene scene, int spp) { return spp >= scene.getTargetSpp(); }
        @Override public boolean saveRenderDump(Scene scene, int spp) { return false; }
      });
      rm.setSceneProvider(new FixedSceneProvider(s));
      rm.start();
      DefaultRenderManager manager = rm;
      Future<?> f = Executors.newSingleThreadExecutor(t -> { Thread th = new Thread(t); th.setDaemon(true); return th; })
          .submit(() -> { manager.join(); return null; });
      f.get(180, TimeUnit.SECONDS);
      r.samples = rm.bufferedScene.getSampleBuffer().clone();
      if (rm.bufferedScene.spp == 0) r.error = "rendered 0 spp";
    } catch (TimeoutException e) {
      r.error = "timeout (renderer hung)";
    } catch (Throwable e) {
      r.error = e.getClass().getSimpleName() + ": " + e.getMessage();
    } finally {
      if (rm != null) rm.shutdown();
    }
    r.millis = System.currentTimeMillis() - t0;
    List<String> newLogs = new ArrayList<>(logLines.subList(logStart, logLines.size()));
    newLogs.removeIf(l -> !l.startsWith("ERROR"));
    if (!newLogs.isEmpty()) {
      String first = newLogs.get(0).replace('\n', ' ').replace('\t', ' ');
      if (first.length() > 200) first = first.substring(0, 200);
      r.error = (r.error == null ? "" : r.error + " | ") + "log " + first;
    }
    if (r.samples != null && allZero(r.samples) && r.error == null && !allowBlack) r.error = "all-black output";
    return r;
  }

  static boolean allZero(double[] a) {
    for (double v : a) if (v != 0) return false;
    return true;
  }

  // ---------------------------------------------------------------- tests

  static void compare(String name, Consumer<Scene> mod) {
    if (skip(name)) return;
    System.out.println("== compare " + name);
    Result cpu = renderCpu(name, mod, true, "PACKED");
    Result gpu = render(GPU, mod, true, "PACKED");
    String result, detail;
    if (gpu.error != null || cpu.error != null) {
      result = "ERROR";
      detail = "cpu=" + cpu.error + " gpu=" + gpu.error;
    } else {
      double[] mc = mean(cpu.samples), mg = mean(gpu.samples);
      double d = blockDiff(cpu.samples, gpu.samples);
      result = d < 0.15 ? "MATCH" : d < 0.35 ? "DIFFERS" : "MISMATCH";
      detail = String.format(Locale.ROOT, "blockRelDiff=%.3f cpuMean=%s gpuMean=%s gpu/cpu=%.2f cpuMs=%d gpuMs=%d",
          d, fmt(mc), fmt(mg), lum(mg) / Math.max(1e-9, lum(mc)), cpu.millis, gpu.millis);
    }
    report.append("compare\t").append(name).append('\t').append(result).append('\t').append(detail).append('\n');
    writeImage(name, cpu, gpu, null, null);
    System.out.println("   " + result + " " + detail);
  }

  static void toggle(String name, Consumer<Scene> a, Consumer<Scene> b) {
    toggle(name, a, b, true, true);
  }

  static void toggle(String name, Consumer<Scene> a, Consumer<Scene> b, boolean waterA, boolean waterB) {
    if (skip(name)) return;
    System.out.println("== toggle " + name);
    Result ca = renderCpu(name + "-A", a, waterA, "PACKED"), cb = renderCpu(name + "-B", b, waterB, "PACKED");
    Result ga = render(GPU, a, waterA, "PACKED"), gb = render(GPU, b, waterB, "PACKED");
    String result, detail;
    if (ca.error != null || cb.error != null || ga.error != null || gb.error != null) {
      result = "ERROR";
      detail = "cpuA=" + ca.error + " cpuB=" + cb.error + " gpuA=" + ga.error + " gpuB=" + gb.error;
    } else {
      double dc = blockDiff(ca.samples, cb.samples), dg = blockDiff(ga.samples, gb.samples);
      boolean cpuReacts = dc > 0.03, gpuReacts = dg > 0.005;
      if (!cpuReacts) result = "N/A(cpu unchanged)";
      else if (!gpuReacts) result = "IGNORED_ON_GPU";
      else if (Math.abs(Math.log((dg + 1e-6) / dc)) > Math.log(2.5)) result = "WEAKER/STRONGER_ON_GPU";
      else result = "SUPPORTED";
      detail = String.format(Locale.ROOT, "cpuChange=%.3f gpuChange=%.4f", dc, dg);
    }
    report.append("toggle\t").append(name).append('\t').append(result).append('\t').append(detail).append('\n');
    writeImage(name, ca, ga, cb, gb);
    System.out.println("   " + result + " " + detail);
  }

  static void robustness(String name, String impl) {
    if (skip(name)) return;
    System.out.println("== robustness " + name);
    Result cpu = renderCpu(name, s -> {}, true, impl);
    Result gpu = render(GPU, s -> {}, true, impl);
    String result = gpu.error != null ? "GPU_FAILS" : "OK";
    String detail = "cpu=" + (cpu.error == null ? "ok" : cpu.error) + " gpu=" + (gpu.error == null ? "ok" : gpu.error);
    report.append("robust\t").append(name).append('\t').append(result).append('\t').append(detail).append('\n');
    writeImage(name, cpu, gpu, null, null);
    System.out.println("   " + result + " " + detail);
  }

  // ---------------------------------------------------------------- metrics & images

  static double lum(double[] c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }

  static double[] mean(double[] s) {
    double[] m = new double[3];
    for (int i = 0; i < s.length; i++) m[i % 3] += s[i];
    for (int i = 0; i < 3; i++) m[i] /= (s.length / 3.0);
    return m;
  }

  static String fmt(double[] c) {
    return String.format(Locale.ROOT, "(%.3f,%.3f,%.3f)", c[0], c[1], c[2]);
  }

  /** Mean absolute luminance difference over 16x16 blocks, relative to the mean luminance. */
  static double blockDiff(double[] a, double[] b) {
    int B = 16, bw = W / B, bh = H / B;
    double diff = 0, norm = 0;
    for (int by = 0; by < bh; by++)
      for (int bx = 0; bx < bw; bx++) {
        double la = 0, lb = 0;
        for (int y = by * B; y < (by + 1) * B; y++)
          for (int x = bx * B; x < (bx + 1) * B; x++) {
            int o = 3 * (y * W + x);
            la += lum(new double[] {a[o], a[o + 1], a[o + 2]});
            lb += lum(new double[] {b[o], b[o + 1], b[o + 2]});
          }
        diff += Math.abs(la - lb);
        norm += (la + lb) / 2;
      }
    return diff / Math.max(1e-9, norm);
  }

  static void writeImage(String name, Result tl, Result tr, Result bl, Result br) {
    int rows = bl == null ? 1 : 2;
    BufferedImage img = new BufferedImage(W * 2 + 4, H * rows + 4 * (rows - 1), BufferedImage.TYPE_INT_RGB);
    paint(img, tl, 0, 0);
    paint(img, tr, W + 4, 0);
    if (bl != null) {
      paint(img, bl, 0, H + 4);
      paint(img, br, W + 4, H + 4);
    }
    try {
      ImageIO.write(img, "png", new File(OUT, name + ".png"));
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  static void paint(BufferedImage img, Result r, int ox, int oy) {
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        int rgb;
        if (r == null || r.samples == null) {
          rgb = ((x / 8 + y / 8) % 2 == 0) ? 0xFF00FF : 0x000000;
        } else {
          int o = 3 * (y * W + x);
          int cr = tone(r.samples[o]), cg = tone(r.samples[o + 1]), cb = tone(r.samples[o + 2]);
          rgb = (cr << 16) | (cg << 8) | cb;
        }
        img.setRGB(ox + x, oy + y, rgb);
      }
  }

  static int tone(double v) {
    return (int) Math.round(255 * Math.pow(Math.min(1, Math.max(0, v)), 1 / 2.2));
  }

  public static class FixedSceneProvider implements SceneProvider {
    private final Scene scene;
    private boolean change = true;

    FixedSceneProvider(Scene scene) { this.scene = scene; }

    @Override public synchronized ResetReason awaitSceneStateChange() throws InterruptedException {
      while (!change) wait();
      change = false;
      return ResetReason.SCENE_LOADED;
    }

    @Override public synchronized boolean pollSceneStateChange() { return change; }

    @Override public synchronized void withSceneProtected(Consumer<Scene> fun) { fun.accept(scene); }

    @Override public synchronized void withEditSceneProtected(Consumer<Scene> fun) {}

    @Override public void addChangeListener(BiConsumer<ResetReason, Scene> listener) {}

    @Override public void removeChangeListener(BiConsumer<ResetReason, Scene> listener) {}
  }
}
