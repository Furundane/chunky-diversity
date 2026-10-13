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
 */
public class SkyData {
    public final int resolution;
    public final byte[] texture;
    /** The kernel's {@code skyIntensity}. */
    public final float intensity;

    public SkyData(Scene scene) {
        this.resolution = getTextureResolution(scene);
        this.intensity = (float) scene.sun().getIntensity();

        texture = new byte[resolution * resolution * 4];
        Ray ray = new Ray();
        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                int offset = 4 * (j * resolution + i);

                double theta = ((double) i / resolution) * 2 * FastMath.PI;
                double phi = ((double) j / resolution) * FastMath.PI - FastMath.PI / 2;
                double r = FastMath.cos(phi);
                ray.d.set(FastMath.cos(theta) * r, FastMath.sin(phi), FastMath.sin(theta) * r);

                scene.sky().getSkyColor(ray, false);
                texture[offset + 0] = (byte) (ray.color.x * 255);
                texture[offset + 1] = (byte) (ray.color.y * 255);
                texture[offset + 2] = (byte) (ray.color.z * 255);
                texture[offset + 3] = (byte) 255;
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
