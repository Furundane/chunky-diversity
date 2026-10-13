package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;

import java.io.*;

/**
 * Renders a scene copy with the GPU renderer and writes the raw sample buffer, for noise/bias comparisons.
 * Args: scene directory, scene name, width, height, spp, output file.
 */
public class NoiseRender {
  public static void main(String[] args) throws Exception {
    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);
    RenderContext ctx = new RenderContext(chunky);
    ctx.setSceneDirectory(new File(args[0]));
    Scene s = new Scene();
    s.loadScene(ctx, args[1], TaskTracker.NONE);
    s.setCanvasSize(Integer.parseInt(args[2]), Integer.parseInt(args[3]));
    s.setRenderer("ChunkyClRenderer");
    s.setTargetSpp(Integer.parseInt(args[4]));
    s.setBufferFinalization(true);
    s.setRenderMode(RenderMode.RENDERING);
    DefaultRenderManager rm = new DefaultRenderManager(ctx, true);
    rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
    long t0 = System.nanoTime();
    rm.start();
    rm.join();
    double sec = (System.nanoTime() - t0) / 1e9;
    double[] samples = rm.bufferedScene.getSampleBuffer();
    try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[5])))) {
      out.writeInt(samples.length);
      for (double v : samples) out.writeDouble(v);
    }
    System.out.printf(java.util.Locale.ROOT, "NOISE spp=%d seconds=%.2f file=%s%n", rm.bufferedScene.spp, sec, args[5]);
    System.exit(0);
  }
}
