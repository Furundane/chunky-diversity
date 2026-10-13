package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.util.*;

/**
 * Renders the harness scene at 1080p with the GPU renderer in GUI conditions (buffer finalization on)
 * for a fixed time and reports where the CPU time goes, per thread group.
 */
public class CpuProfile {
  public static void main(String[] args) throws Exception {
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) { System.out.println("[log " + level + "] " + message); }
    }, Level.ERROR, Level.WARNING);
    int seconds = args.length > 0 ? Integer.parseInt(args[0]) : 15;
    String label = args.length > 1 ? args[1] : "run";

    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);

    Scene s = FeatureHarness.buildScene("PACKED", true);
    s.setCanvasSize(1920, 1080);
    s.setRenderer("ChunkyClRenderer");
    s.setTargetSpp(1_000_000);
    s.setRenderMode(RenderMode.RENDERING);
    s.setBufferFinalization(true); // canvas visible, like the GUI

    ThreadMXBean mx = ManagementFactory.getThreadMXBean();
    com.sun.management.OperatingSystemMXBean os =
        (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

    DefaultRenderManager rm = new DefaultRenderManager(new RenderContext(chunky), true);
    rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
    rm.start();

    // Warm up (scene upload, kernel build), then measure.
    Thread.sleep(3000);
    Map<Long, Long> cpu0 = threadCpu(mx);
    long proc0 = os.getProcessCpuTime();
    int spp0 = rm.bufferedScene.spp;
    long t0 = System.nanoTime();

    Thread.sleep(seconds * 1000L);

    Map<Long, Long> cpu1 = threadCpu(mx);
    long proc1 = os.getProcessCpuTime();
    int spp1 = rm.bufferedScene.spp;
    double wall = (System.nanoTime() - t0) / 1e9;

    Map<String, Double> byGroup = new TreeMap<>();
    for (Map.Entry<Long, Long> e : cpu1.entrySet()) {
      ThreadInfo info = mx.getThreadInfo(e.getKey());
      if (info == null) continue;
      long before = cpu0.getOrDefault(e.getKey(), 0L);
      String group = info.getThreadName().replaceAll("[-#]?\\d+$", "").replaceAll("-\\d+-", "-N-");
      byGroup.merge(group, (e.getValue() - before) / 1e9, Double::sum);
    }

    System.out.printf(Locale.ROOT, "== %s: %.1fs wall, %d cores%n", label, wall, Runtime.getRuntime().availableProcessors());
    System.out.printf(Locale.ROOT, "process CPU: %.1f core-seconds = %.2f cores busy on average%n",
        (proc1 - proc0) / 1e9, (proc1 - proc0) / 1e9 / wall);
    System.out.printf(Locale.ROOT, "samples: %d spp in %.1fs = %.1f spp/s (1920x1080)%n", spp1 - spp0, wall, (spp1 - spp0) / wall);
    byGroup.entrySet().stream()
        .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
        .limit(8)
        .forEach(e -> System.out.printf(Locale.ROOT, "  %-45s %6.2f cores%n", e.getKey(), e.getValue() / wall));
    rm.shutdown();
    System.exit(0);
  }

  static Map<Long, Long> threadCpu(ThreadMXBean mx) {
    Map<Long, Long> out = new HashMap<>();
    for (long id : mx.getAllThreadIds()) {
      long t = mx.getThreadCpuTime(id);
      if (t >= 0) out.put(id, t);
    }
    return out;
  }
}
