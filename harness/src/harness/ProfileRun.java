package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;
import se.llbit.util.TaskTracker;

import java.io.File;

/**
 * Renders a scene with the (instrumented) GPU renderer for a fixed number of passes; the instrumented
 * renderer prints a PROFILE line. Args: "synthetic" | scene directory + scene name, width, height, spp.
 */
public class ProfileRun {
  public static void main(String[] args) throws Exception {
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) {
        if (level == Level.ERROR) System.out.println("[log " + level + "] " + message);
      }
    }, Level.ERROR, Level.WARNING);
    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);

    int i = 0;
    RenderContext ctx = new RenderContext(chunky);
    Scene s;
    if (args[0].equals("synthetic")) {
      s = FeatureHarness.buildScene("PACKED", true);
      i = 1;
    } else {
      ctx.setSceneDirectory(new File(args[0]));
      s = new Scene();
      s.loadScene(ctx, args[1], TaskTracker.NONE);
      i = 2;
    }
    s.setCanvasSize(Integer.parseInt(args[i]), Integer.parseInt(args[i + 1]));
    if (args.length > i + 3) s.setRayDepth(Integer.parseInt(args[i + 3]));
    s.setRenderer(FeatureHarness.GPU);
    s.setTargetSpp(Integer.parseInt(args[i + 2]));
    s.setBufferFinalization(true);
    s.setRenderMode(RenderMode.RENDERING);

    DefaultRenderManager rm = new DefaultRenderManager(ctx, true);
    rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
    long t0 = System.nanoTime();
    rm.start();
    rm.join();
    double ms = (System.nanoTime() - t0) / 1e6;
    System.out.printf(java.util.Locale.ROOT, "TIMING spp=%d totalMs=%.0f msPerSpp=%.2f rayDepth=%d%n",
        rm.bufferedScene.spp, ms, ms / rm.bufferedScene.spp, s.getRayDepth());
    System.exit(0);
  }
}
