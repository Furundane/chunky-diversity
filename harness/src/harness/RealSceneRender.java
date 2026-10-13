package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.resources.BitmapImage;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;
import se.llbit.util.TaskTracker;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Renders a copy of a real Chunky scene with the CPU path tracer and with ChunkyCL and saves both
 * post-processed images. Args: scene directory, scene name, width, height, cpu spp, gpu spp, output prefix.
 */
public class RealSceneRender {
  public static void main(String[] args) throws Exception {
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) { System.out.println("[log " + level + "] " + message); }
    }, Level.ERROR, Level.WARNING);
    File dir = new File(args[0]);
    String name = args[1];
    int width = Integer.parseInt(args[2]), height = Integer.parseInt(args[3]);
    int cpuSpp = Integer.parseInt(args[4]), gpuSpp = Integer.parseInt(args[5]);
    String prefix = args[6];

    Chunky.loadDefaultTextures();
    ChunkyOptions options = ChunkyOptions.getDefaults();
    options.renderThreads = Runtime.getRuntime().availableProcessors();
    Chunky chunky = new Chunky(options);
    new ChunkyCl().attach(chunky);

    for (String renderer : new String[] {"ChunkyClRenderer", DefaultRenderManager.ChunkyPathTracerID}) {
      boolean gpu = renderer.equals("ChunkyClRenderer");
      if (!gpu && cpuSpp <= 0) continue;
      RenderContext ctx = new RenderContext(chunky);
      ctx.setSceneDirectory(dir);
      Scene s = new Scene();
      s.loadScene(ctx, name, TaskTracker.NONE);
      s.setCanvasSize(width, height);
      s.setRenderer(renderer);
      s.setTargetSpp(gpu ? gpuSpp : cpuSpp);
      s.setBufferFinalization(true);
      s.setRenderMode(RenderMode.RENDERING);

      DefaultRenderManager rm = new DefaultRenderManager(ctx, true);
      rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
      long t0 = System.currentTimeMillis();
      rm.start();
      rm.join();
      long ms = System.currentTimeMillis() - t0;

      Scene done = rm.bufferedScene;
      done.postProcessFrame(TaskTracker.Task.NONE);
      BitmapImage bitmap = done.getBackBuffer();
      BufferedImage img = new BufferedImage(bitmap.width, bitmap.height, BufferedImage.TYPE_INT_RGB);
      for (int y = 0; y < bitmap.height; y++)
        for (int x = 0; x < bitmap.width; x++) img.setRGB(x, y, bitmap.getPixel(x, y));
      File out = new File("out/" + prefix + (gpu ? "_gpu" : "_cpu") + ".png");
      ImageIO.write(img, "png", out);
      System.out.printf("%s: %d spp in %d ms -> %s%n", renderer, done.spp, ms, out);
      rm.shutdown();
    }
    System.exit(0);
  }
}
