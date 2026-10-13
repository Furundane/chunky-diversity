package harness;

import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.renderer.RenderContext;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.math.Octree;
import se.llbit.math.PackedOctree;
import se.llbit.util.TaskTracker;

import java.io.File;

/** Prints the octree implementation and size of saved scenes. Args: scene directory + scene name, repeated. */
public class OctreeInfo {
  public static void main(String[] args) throws Exception {
    Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
    for (int i = 0; i + 1 < args.length; i += 2) {
      RenderContext ctx = new RenderContext(chunky);
      ctx.setSceneDirectory(new File(args[i]));
      Scene s = new Scene();
      s.loadScene(ctx, args[i + 1], TaskTracker.NONE);
      System.out.printf("%s: chunks=%d world %s, water %s%n", args[i + 1], s.numberOfChunks(),
          describe(s.getWorldOctree()), describe(s.getWaterOctree()));
    }
    System.exit(0);
  }

  static String describe(Octree octree) {
    Octree.OctreeImplementation impl = octree.getImplementation();
    String size = impl instanceof PackedOctree
        ? String.format("%.1f MB", ((PackedOctree) impl).treeData.length * 4 / 1e6)
        : "n/a";
    return impl.getClass().getSimpleName() + " depth " + impl.getDepth() + " " + size;
  }
}
