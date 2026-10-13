package harness;

import dev.thatredox.chunkynative.opencl.ChunkyCl;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/** Renders the harness scene once with a preview renderer and saves the preview image. */
public class PreviewDebug {
  public static void main(String[] args) throws Exception {
    Log.setReceiver(new Receiver() {
      @Override public void logEvent(Level level, String message) { System.out.println("[log " + level + "] " + message); }
    }, Level.ERROR, Level.WARNING);
    Chunky.loadDefaultTextures();
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    new ChunkyCl().attach(chunky);

    for (String id : new String[] {DefaultRenderManager.ChunkyPreviewID, "ChunkyClPreviewRenderer"}) {
      Scene s = FeatureHarness.buildScene("PACKED", true);
      s.setRenderMode(RenderMode.PREVIEW);
      s.setPreviewRenderer(id);
      s.setBufferFinalization(true); // previews only render while the canvas is visible
      DefaultRenderManager rm = new DefaultRenderManager(new RenderContext(chunky), true);
      rm.setSceneProvider(new FeatureHarness.FixedSceneProvider(s));
      rm.start();
      Thread.sleep(4000);
      int[] w = new int[1], h = new int[1];
      BufferedImage[] out = new BufferedImage[1];
      // Headless there is no canvas to trigger buffer swaps, so take whichever buffer holds the frame.
      se.llbit.chunky.resources.BitmapImage[] bitmaps = new se.llbit.chunky.resources.BitmapImage[2];
      rm.withBufferedImage(bitmap -> bitmaps[0] = bitmap);
      bitmaps[1] = rm.bufferedScene.getBackBuffer();
      for (se.llbit.chunky.resources.BitmapImage bitmap : bitmaps) {
        if (bitmap == null) continue;
        BufferedImage img = new BufferedImage(bitmap.width, bitmap.height, BufferedImage.TYPE_INT_RGB);
        long sum = 0;
        for (int y = 0; y < bitmap.height; y++)
          for (int x = 0; x < bitmap.width; x++) {
            img.setRGB(x, y, bitmap.getPixel(x, y));
            sum += bitmap.getPixel(x, y) & 0xFFFFFF;
          }
        if (out[0] == null || sum > 0) out[0] = img;
      }
      rm.shutdown();
      ImageIO.write(out[0], "png", new File("out/preview_" + id + ".png"));
      System.out.println("saved preview_" + id + ".png");
    }
    System.exit(0);
  }
}
