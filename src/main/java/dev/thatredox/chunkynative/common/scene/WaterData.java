package dev.thatredox.chunkynative.common.scene;

import se.llbit.chunky.renderer.SunSamplingStrategy;
import se.llbit.chunky.model.minecraft.WaterModel;
import se.llbit.chunky.renderer.scene.LegacyWaterShader;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.SimplexWaterShader;
import se.llbit.chunky.renderer.scene.WaterShader;
import se.llbit.chunky.resources.Texture;
import se.llbit.math.Vector3;

import java.lang.reflect.Field;
import java.util.Arrays;

/**
 * Water settings for the kernel (see {@code WaterSettings} in water.h):
 * color (3 floats, linear), opacity, visibility, index of refraction, use custom color, and the scene
 * translucency settings (transmissivity cap, fancier translucency) that other translucent materials use too,
 * the sun sampling strategy flags ({@code SUN_*} in kernel.h), the emitter intensity (0 when emitters
 * are disabled), and the average color of the water texture (3 floats, linear), which Chunky uses for water
 * surfaces instead of the texture itself.
 *
 * <p>Then the water waves (Chunky's WaterShader): 14 mode (0 still, 1 simplex noise, 2 tiled normal map),
 * 15 - 19 simplex parameters (iterations, base frequency, base amplitude, animation speed, animation
 * time), 20 normal map width and from 21 the normal map (2 floats per texel, WaterModel.normalMap).
 */
public class WaterData {
    private static final int WAVES = 14;

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
        float[] data = {
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
        return withWaves(data, scene);
    }

    private static float[] withWaves(float[] base, Scene scene) {
        WaterShader shader = scene.getCurrentWaterShader();
        float[] normalMap = shader instanceof LegacyWaterShader ? waterNormalMap() : new float[0];
        float[] out = Arrays.copyOf(base, WAVES + 7 + normalMap.length);
        if (shader instanceof SimplexWaterShader) {
            SimplexWaterShader simplex = (SimplexWaterShader) shader;
            out[WAVES] = 1;
            out[WAVES + 1] = simplex.iterations;
            out[WAVES + 2] = (float) simplex.baseFrequency;
            out[WAVES + 3] = (float) simplex.baseAmplitude;
            out[WAVES + 4] = (float) simplex.animationSpeed;
        } else if (shader instanceof LegacyWaterShader) {
            out[WAVES] = 2;
        }
        out[WAVES + 5] = (float) scene.getAnimationTime();
        out[WAVES + 6] = (float) Math.sqrt(normalMap.length / 2.0);
        System.arraycopy(normalMap, 0, out, WAVES + 7, normalMap.length);
        return out;
    }

    /** WaterModel.normalMap: derivatives of the water height texture, 2 floats per texel. */
    private static float[] waterNormalMap() {
        try {
            Field field = WaterModel.class.getDeclaredField("normalMap");
            field.setAccessible(true);
            return (float[]) field.get(null);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Could not read the water normal map", e);
        }
    }

    private static int sunStrategyFlags(SunSamplingStrategy strategy) {
        return (strategy.doSunSampling() ? 1 : 0)
                | (strategy.isDiffuseSun() ? 2 : 0)
                | (strategy.isStrictDirectLight() ? 4 : 0)
                | (strategy.isSunLuminosity() ? 8 : 0);
    }
}
