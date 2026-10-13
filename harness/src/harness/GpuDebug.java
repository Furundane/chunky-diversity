package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import dev.thatredox.chunkynative.opencl.context.ContextManager;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;

import java.util.Arrays;

/** Runs one GPU render and prints internal state, to tell harness bugs from plugin bugs. */
public class GpuDebug {
  public static void main(String[] args) throws Exception {
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) { System.out.println("[log " + level + "] " + message); }
      @Override public void logEvent(Level level, String message, Throwable t) { System.out.println("[log " + level + "] " + message); t.printStackTrace(System.out); }
      @Override public void logEvent(Level level, Throwable t) { System.out.println("[log " + level + "]"); t.printStackTrace(System.out); }
    }, Level.ERROR, Level.WARNING, Level.INFO);
    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);
    System.out.println("device: " + ContextManager.get().device.name() + " / " + ContextManager.get().device.type());
    System.out.println("renderers: " + DefaultRenderManager.renderers.keySet());

    Scene s = FeatureHarness.buildScene("PACKED", true);
    s.setRenderer(args.length > 0 ? args[0] : FeatureHarness.GPU);
    s.setTargetSpp(64);
    RenderContext ctx = new RenderContext(chunky);
    DefaultRenderManager rm = new DefaultRenderManager(ctx, true);
    rm.setSnapshotControl(new SnapshotControl() {
      @Override public boolean saveSnapshot(Scene scene, int spp) {
        System.out.println("saveSnapshot? spp=" + spp + " target=" + scene.getTargetSpp());
        return spp >= scene.getTargetSpp();
      }
      @Override public boolean saveRenderDump(Scene scene, int spp) { return false; }
    });
    rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
    long t0 = System.currentTimeMillis();
    rm.start();
    rm.join();
    double[] buf = rm.bufferedScene.getSampleBuffer();
    System.out.println("done in " + (System.currentTimeMillis() - t0) + "ms, spp=" + rm.bufferedScene.spp
        + " renderer=" + rm.bufferedScene.getRenderer()
        + " max=" + Arrays.stream(buf).max().getAsDouble() + " mean=" + Arrays.stream(buf).average().getAsDouble());
    System.out.println("first values: " + Arrays.toString(Arrays.copyOf(buf, 12)));
    System.exit(0);
  }
}
