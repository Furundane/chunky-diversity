package dev.thatredox.chunkynative.common.scene;

import se.llbit.chunky.renderer.SunSamplingStrategy;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.resources.Texture;
import se.llbit.math.Vector3;

/**
 * Water settings for the kernel (see {@code WaterSettings} in water.h):
 * color (3 floats, linear), opacity, visibility, index of refraction, use custom color, and the scene
 * translucency settings (transmissivity cap, fancier translucency) that other translucent materials use too,
 * the sun sampling strategy flags ({@code SUN_*} in kernel.h), the emitter intensity (0 when emitters
 * are disabled), and the average color of the water texture (3 floats, linear), which Chunky uses for water
 * surfaces instead of the texture itself.
 */
public class WaterData {
    private WaterData() {}

    public static float[] pack(Scene scene) {
        float[] waterTexture = Texture.water.getAvgColorLinear();
        float[] color;
        if (scene.getUseCustomWaterColor()) {
            Vector3 c = scene.getWaterColor();
            color = new float[] {(float) c.x, (float) c.y, (float) c.z};
        } else {
            // The kernel has no biome data, so use the biome water color at the camera.
            Vector3 pos = new Vector3(scene.camera().getPosition());
            pos.sub(scene.getOrigin());
            color = scene.getWaterColor((int) Math.floor(pos.x), (int) Math.floor(pos.y), (int) Math.floor(pos.z));
        }
        return new float[] {
                color[0], color[1], color[2],
                (float) scene.getWaterOpacity(),
                (float) scene.getWaterVisibility(),
                scene.getPalette().water.ior,
                scene.getUseCustomWaterColor() ? 1 : 0,
                (float) scene.getTransmissivityCap(),
                scene.getFancierTranslucency() ? 1 : 0,
                sunStrategyFlags(scene.getSunSamplingStrategy()),
                scene.getEmittersEnabled() ? (float) scene.getEmitterIntensity() : 0,
                waterTexture[0], waterTexture[1], waterTexture[2],
        };
    }

    private static int sunStrategyFlags(SunSamplingStrategy strategy) {
        return (strategy.doSunSampling() ? 1 : 0)
                | (strategy.isDiffuseSun() ? 2 : 0)
                | (strategy.isStrictDirectLight() ? 4 : 0)
                | (strategy.isSunLuminosity() ? 8 : 0);
    }
}
