package dev.thatredox.chunkynative.common.scene;

import org.apache.commons.math3.util.FastMath;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.chunky.renderer.scene.sky.SkyCache;
import se.llbit.log.Log;
import se.llbit.math.Ray;

import java.lang.reflect.Field;

/**
 * The sky baked into a square float image (equirectangular: x is the azimuth, y the elevation) for the
 * kernel's {@code skyTexture} argument: the resolution, then linear RGB per texel, row by row. Floats
 * keep skies brighter than 1 (HDR). The kernel interpolates bilinearly (Sky_intersect in sky.h).
 *
 * <p>The texture holds the sky color times the sky exposure. The kernel multiplies it by
 * {@link #skyLight}, like Chunky's {@code Sky.getSkyColor} (reflected rays) and
 * {@code Sky.getSkyColorInterpolated} (camera rays). The sun intensity does not apply to the sky.
 */
public class SkyData {
    public final int resolution;
    public final float[] texture;
    /** The kernel's {@code skyLight}: the sky light modifier and the apparent sky light modifier. */
    public final float[] skyLight;

    public SkyData(Scene scene) {
        this.resolution = getTextureResolution(scene);
        Sky sky = scene.sky();
        double skyLightModifier = sky.getSkyLight();
        double apparentSkyLight = sky.getApparentSkyLight();
        this.skyLight = new float[] {(float) skyLightModifier, (float) apparentSkyLight};

        texture = new float[1 + resolution * resolution * 3];
        texture[0] = resolution;
        Ray ray = new Ray();
        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                int offset = 1 + 3 * (j * resolution + i);

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
                texture[offset + 0] = (float) (ray.color.x * scale);
                texture[offset + 1] = (float) (ray.color.y * scale);
                texture[offset + 2] = (float) (ray.color.z * scale);
            }
        }
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
