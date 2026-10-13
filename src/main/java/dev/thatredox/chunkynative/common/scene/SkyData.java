package dev.thatredox.chunkynative.common.scene;

import org.apache.commons.math3.util.FastMath;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.chunky.renderer.scene.sky.SkyCache;
import se.llbit.log.Log;
import se.llbit.math.Ray;

import java.lang.reflect.Field;

/**
 * The sky baked into a square RGBA8 texture (equirectangular: x is the azimuth, y the elevation),
 * for the kernel's {@code skyTexture} argument, sampled with linear filtering.
 *
 * <p>The texture holds the sky color times the sky exposure. The kernel multiplies it by
 * {@link #skyLight}, like Chunky's {@code Sky.getSkyColor} (reflected rays) and
 * {@code Sky.getSkyColorInterpolated} (camera rays). The sun intensity does not apply to the sky.
 */
public class SkyData {
    public final int resolution;
    public final byte[] texture;
    /** The kernel's {@code skyLight}: the sky light modifier and the apparent sky light modifier. */
    public final float[] skyLight;

    public SkyData(Scene scene) {
        this.resolution = getTextureResolution(scene);
        Sky sky = scene.sky();
        double skyLightModifier = sky.getSkyLight();
        double apparentSkyLight = sky.getApparentSkyLight();
        this.skyLight = new float[] {(float) skyLightModifier, (float) apparentSkyLight};

        texture = new byte[resolution * resolution * 4];
        Ray ray = new Ray();
        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                int offset = 4 * (j * resolution + i);

                double theta = ((double) i / resolution) * 2 * FastMath.PI;
                double phi = ((double) j / resolution) * FastMath.PI - FastMath.PI / 2;
                double r = FastMath.cos(phi);
                ray.d.set(FastMath.cos(theta) * r, FastMath.sin(phi), FastMath.sin(theta) * r);

                // Chunky only exposes the sky color with one of the two modifiers applied; divide it out.
                double scale;
                if (skyLightModifier > 0) {
                    sky.getSkyColor(ray, false);
                    scale = 1 / skyLightModifier;
                } else if (apparentSkyLight > 0) {
                    sky.getApparentSkyColor(ray, false);
                    scale = 1 / apparentSkyLight;
                } else {
                    ray.color.set(0, 0, 0, 1);
                    scale = 0;
                }
                texture[offset + 0] = toByte(ray.color.x * scale);
                texture[offset + 1] = toByte(ray.color.y * scale);
                texture[offset + 2] = toByte(ray.color.z * scale);
                texture[offset + 3] = (byte) 255;
            }
        }
    }

    /** Clamped to [0, 1]: casting a larger value to byte used to wrap around to a dark color. */
    private static byte toByte(double value) {
        return (byte) Math.max(0, Math.min(255, (int) (value * 255)));
    }

    private static int getTextureResolution(Scene scene) {
        try {
            Sky sky = scene.sky();
            Field skyCacheField = sky.getClass().getDeclaredField("skyCache");
            skyCacheField.setAccessible(true);
            SkyCache skyCache = (SkyCache) skyCacheField.get(sky);

            return skyCache.getSkyResolution();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            Log.error(e);
            throw new RuntimeException();
        }
    }
}
