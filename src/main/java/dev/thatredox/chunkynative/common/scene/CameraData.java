package dev.thatredox.chunkynative.common.scene;

import dev.thatredox.chunkynative.util.Reflection;
import dev.thatredox.chunkynative.util.Util;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.floats.FloatList;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.renderer.scene.Camera;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.math.Matrix3;
import se.llbit.math.Ray;
import se.llbit.math.Vector3;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Camera data for the kernel's {@code projectorType} and {@code cameraSettings} arguments. The
 * pinhole projection is computed on the GPU from a few settings; other projections use rays
 * generated on the CPU ({@link #generateRays}).
 */
public class CameraData {
    /** Kernel projector type: 0 for pinhole, -1 for pre-generated rays. */
    public final int projectorType;
    /** Pinhole settings: position, transform, DoF, subject distance, field of view. Empty for pre-generated rays. */
    public final float[] settings;

    public CameraData(Scene scene) {
        Camera camera = scene.camera();

        int projType = -1;
        Vector3 pos = new Vector3(camera.getPosition());
        pos.sub(scene.getOrigin());

        FloatArrayList settings = new FloatArrayList();
        settings.addAll(FloatList.of(Util.vector3ToFloat(pos)));
        settings.addAll(FloatList.of(Util.matrix3ToFloat(Reflection.getFieldValue(camera, "transform", Matrix3.class))));

        switch (camera.getProjectionMode()) {
            case PINHOLE:
                projType = 0;
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) Camera.clampedFovTan(camera.getFov()));
                break;
            default:
                // We need to pre-generate rays
                break;
        }

        this.projectorType = projType;
        this.settings = projType == -1 ? new float[0] : settings.toFloatArray();
    }

    public boolean needsRayGeneration() {
        return projectorType == -1;
    }

    /** Size of the pre-generated ray buffer: origin and direction for every pixel. */
    public static int rayBufferFloats(Scene scene) {
        return scene.canvasConfig.getWidth() * scene.canvasConfig.getHeight() * 3 * 2;
    }

    /** Generates one camera ray (origin, direction) per pixel, jittered within the pixel. */
    public static float[] generateRays(Scene scene, boolean jitter) {
        int width = scene.canvasConfig.getWidth();
        int height = scene.canvasConfig.getHeight();
        int fullWidth = scene.canvasConfig.getCropWidth();
        int fullHeight = scene.canvasConfig.getCropHeight();
        int cropX = scene.canvasConfig.getCropX();
        int cropY = scene.canvasConfig.getCropY();

        float[] rays = new float[rayBufferFloats(scene)];

        double halfWidth = fullWidth / (2.0 * fullHeight);
        double invHeight = 1.0 / fullHeight;

        Camera cam = scene.camera();

        Chunky.getCommonThreads().submit(() -> IntStream.range(0, width).parallel().forEach(i -> {
            Ray ray = new Ray();
            Random random = jitter ? ThreadLocalRandom.current() : null;
            for (int j = 0; j < height; j++) {
                int offset = (j * width + i) * 3 * 2;

                float ox = jitter ? random.nextFloat(): 0.5f;
                float oy = jitter ? random.nextFloat(): 0.5f;

                double x = -halfWidth + (i + ox + cropX) * invHeight;
                double y = -0.5 + (j + oy + cropY) * invHeight;
                // The overload with a Random applies depth of field; the one without does not.
                if (jitter) {
                    cam.calcViewRay(ray, random, x, y);
                } else {
                    cam.calcViewRay(ray, x, y);
                }
                ray.o.sub(scene.getOrigin());

                System.arraycopy(Util.vector3ToFloat(ray.o), 0, rays, offset, 3);
                System.arraycopy(Util.vector3ToFloat(ray.d), 0, rays, offset+3, 3);
            }
        })).join();
        return rays;
    }

    /** The kernel's {@code canvasConfig}: width, height, full (crop) width and height, crop x and y. */
    public static int[] canvasConfig(Scene scene) {
        return new int[] {
                scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        };
    }
}
