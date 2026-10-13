package harness;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.context.Device;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;

/** Steps through OpenCL setup one call at a time, to locate failures on a new platform. */
public class ClProbe {
  public static void main(String[] args) {
    Device[] devices = Device.getDevices();
    for (Device d : devices) {
      System.out.println("device " + d.id + ": " + d.name() + " / " + d.type() + " / " + d.versionString());
    }
    Device d = devices[0];
    System.out.println("creating context on " + d.name());
    ClContext ctx = new ClContext(d);
    System.out.println("context ok; building tonemap");
    KernelLoader.loadProgram(ctx, "tonemap", "post_processing_filter.c");
    System.out.println("tonemap ok; building kernel");
    KernelLoader.loadProgram(ctx, "kernel", "rayTracer.c");
    System.out.println("kernel ok");
  }
}
