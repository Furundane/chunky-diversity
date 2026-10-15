package harness;

/**
 * Experiment: the MetalFX temporal denoiser (harness/metalfx/metalfx_denoiser.m). The library is loaded
 * from -Dharness.metalfxLib (built by harness/metalfx/build.sh).
 */
public final class MetalFxDenoiser implements AutoCloseable {
  static {
    System.load(System.getProperty("harness.metalfxLib"));
  }

  private static native long create(int width, int height, int outWidth, int outHeight);

  private static native long denoise(long handle, float[] color, float[] depth, float[] motion, float[] normal,
                                     float[] diffuse, float[] specular, float[] roughness, float[] params, float[] out);

  private static native void release(long handle);

  private final long handle;

  public MetalFxDenoiser(int width, int height, int outWidth, int outHeight) {
    handle = create(width, height, outWidth, outHeight);
    if (handle == 0) throw new IllegalStateException("MetalFX denoiser unavailable");
  }

  /**
   * Denoises one frame and returns the GPU time of the denoiser in nanoseconds. params: jitter x, y,
   * reset history (0/1), world-to-view and view-to-clip matrices (16 floats each, column-major).
   */
  public long denoise(float[] color, float[] depth, float[] motion, float[] normal, float[] diffuse,
                      float[] specular, float[] roughness, float[] params, float[] out) {
    long nanos = denoise(handle, color, depth, motion, normal, diffuse, specular, roughness, params, out);
    if (nanos < 0) throw new IllegalStateException("MetalFX denoise failed");
    return nanos;
  }

  @Override
  public void close() {
    release(handle);
  }
}
