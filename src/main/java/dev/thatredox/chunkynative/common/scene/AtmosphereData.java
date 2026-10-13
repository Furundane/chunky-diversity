package dev.thatredox.chunkynative.common.scene;

import dev.thatredox.chunkynative.util.Reflection;
import se.llbit.chunky.renderer.scene.Fog;
import se.llbit.chunky.renderer.scene.FogLayer;
import se.llbit.chunky.renderer.scene.FogMode;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.chunky.world.Clouds;
import se.llbit.math.Vector3i;

import java.util.List;

/**
 * Fog and clouds for the kernel's {@code atmosphere} argument (see atmosphere.h), as ints; floats are
 * stored as their bits.
 * <pre>
 *   0      fog mode: 0 none, 1 uniform, 2 layered
 *   1      float uniform density
 *   2      float sky fog density
 *   3      fast fog (0 or 1)
 *   4 - 6  float fog color
 *   7, 8   float scene y min and y max (used by Chunky for the sky fog scatter point)
 *   9      number of fog layers (at most MAX_FOG_LAYERS)
 *   10     clouds enabled (0 or 1)
 *   11     float cloud size
 *   12-14  float cloud offset (x, z in cloud cells, y in blocks)
 *   15-17  float scene origin (clouds are placed in world coordinates)
 *   32 + 4 * i: fog layer i: float y (relative to y min), breadth, 1 / breadth, density
 *   CLOUD_BITS: the 256 x 256 cloud bitmap, 32 cells per int, row by row (z)
 * </pre>
 */
public class AtmosphereData {
    public static final int MAX_FOG_LAYERS = 16;
    public static final int FOG_LAYERS = 32;
    public static final int CLOUD_BITS = FOG_LAYERS + 4 * MAX_FOG_LAYERS;
    public static final int SIZE = CLOUD_BITS + 256 * 256 / 32;

    private AtmosphereData() {}

    /**
     * Preprocessor defines for the render kernel variant this scene needs. The fog and cloud code is
     * only compiled in when used: even when skipped at runtime it slows the kernel down (by about 6%).
     */
    public static String[] kernelDefines(Scene scene) {
        boolean fog = scene.fog.fogEnabled();
        boolean clouds = scene.sky().cloudsEnabled();
        if (fog && clouds) return new String[] {"CHUNKYCL_FOG", "CHUNKYCL_CLOUDS"};
        if (fog) return new String[] {"CHUNKYCL_FOG"};
        if (clouds) return new String[] {"CHUNKYCL_CLOUDS"};
        return new String[0];
    }

    public static int[] pack(Scene scene) {
        int[] out = new int[SIZE];
        Fog fog = scene.fog;
        FogMode mode = fog.getFogMode();
        out[0] = mode == FogMode.UNIFORM ? 1 : mode == FogMode.LAYERED ? 2 : 0;
        out[1] = bits(fog.getUniformDensity());
        out[2] = bits(fog.getSkyFogDensity());
        out[3] = fog.fastFog() ? 1 : 0;
        out[4] = bits(fog.getFogColor().x);
        out[5] = bits(fog.getFogColor().y);
        out[6] = bits(fog.getFogColor().z);
        out[7] = bits(Reflection.getFieldValue(scene, "yMin", Integer.class));
        out[8] = bits(Reflection.getFieldValue(scene, "yMax", Integer.class));
        List<FogLayer> layers = fog.getFogLayers();
        int layerCount = Math.min(layers.size(), MAX_FOG_LAYERS);
        out[9] = layerCount;
        for (int i = 0; i < layerCount; i++) {
            FogLayer layer = layers.get(i);
            out[FOG_LAYERS + 4 * i] = bits(layer.yWithOrigin);
            out[FOG_LAYERS + 4 * i + 1] = bits(layer.breadth);
            out[FOG_LAYERS + 4 * i + 2] = bits(layer.breadthInv);
            out[FOG_LAYERS + 4 * i + 3] = bits(layer.density);
        }

        Sky sky = scene.sky();
        out[10] = sky.cloudsEnabled() ? 1 : 0;
        out[11] = bits(sky.cloudSize());
        out[12] = bits(sky.cloudXOffset());
        out[13] = bits(sky.cloudYOffset());
        out[14] = bits(sky.cloudZOffset());
        Vector3i origin = scene.getOrigin();
        out[15] = bits(origin.x);
        out[16] = bits(origin.y);
        out[17] = bits(origin.z);
        if (sky.cloudsEnabled()) {
            for (int z = 0; z < 256; z++) {
                for (int x = 0; x < 256; x++) {
                    if (Clouds.getCloud(x, z) != 0) {
                        int bit = z * 256 + x;
                        out[CLOUD_BITS + bit / 32] |= 1 << (bit % 32);
                    }
                }
            }
        }
        return out;
    }

    private static int bits(double value) {
        return Float.floatToIntBits((float) value);
    }
}
