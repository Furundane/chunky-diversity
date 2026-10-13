package dev.thatredox.chunkynative.opencl.renderer.scene;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import se.llbit.chunky.renderer.SunSamplingStrategy;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.math.Vector3;

import static org.jocl.CL.*;

/**
 * Water settings for the kernel (see {@code WaterSettings} in water.h):
 * color (3 floats, linear), opacity, visibility, index of refraction, use custom color, and the scene
 * translucency settings (transmissivity cap, fancier translucency) that other translucent materials use too,
 * and the sun sampling strategy flags ({@code SUN_*} in kernel.h).
 */
public class ClWater implements AutoCloseable {
    public final ClMemory settings;

    public ClWater(Scene scene, ClContext context) {
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
        };
        settings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * data.length, Pointer.to(data), null));
    }

    private static int sunStrategyFlags(SunSamplingStrategy strategy) {
        return (strategy.doSunSampling() ? 1 : 0)
                | (strategy.isDiffuseSun() ? 2 : 0)
                | (strategy.isStrictDirectLight() ? 4 : 0)
                | (strategy.isSunLuminosity() ? 8 : 0);
    }

    @Override
    public void close() {
        settings.close();
    }
}
