package dev.thatredox.chunkynative.common.export.primitives;

import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.export.Packer;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.math3.util.FastMath;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sun;
import se.llbit.math.Vector3;

public class PackedSun implements Packer {
    public final int flags;
    public final long texture;
    public final float altitude;
    public final float azimuth;
    public final float luminosity;
    public final float radius;
    public final float[] color;
    public final float[] apparentTextureBrightness;
    public final float[] emittance;
    public final float luminosityPdf;

    public PackedSun(Sun sun, AbstractTextureLoader texturePalette) {
        flags = sun.drawTexture() ? 1 : 0;
        texture = texturePalette.get(Sun.texture).get();
        altitude = (float) sun.getAltitude();
        azimuth = (float) sun.getAzimuth();
        luminosity = (float) sun.getLuminosity();
        radius = (float) sun.radius;
        color = toFloats(sun.getColor());
        // As computed in Sun.initSun
        Vector3 apparent = sun.getEnableTextureModification() ? new Vector3(sun.getApparentColor()) : new Vector3(1, 1, 1);
        apparent.scale(FastMath.pow(sun.getApparentBrightness(), Scene.DEFAULT_GAMMA));
        apparentTextureBrightness = toFloats(apparent);
        emittance = toFloats(sun.getEmittance());
        luminosityPdf = (float) sun.getLuminosityPdf();
    }

    private static float[] toFloats(Vector3 v) {
        return new float[] {(float) v.x, (float) v.y, (float) v.z};
    }

    /**
     * Pack the sun into 17 ints.
     * 0: Flags. 1 if the sun texture should be drawn. 0 if not.
     * 1 & 2: Sun texture reference.
     * 3: float sun altitude
     * 4: float sun azimuth
     * 5: float sun luminosity
     * 6: float sun radius (radians)
     * 7 - 9: float sun color, for the sun as seen by diffuse rays (Sun.intersectDiffuse)
     * 10 - 12: float apparent texture brightness, for the sun as seen directly (Sun.intersect)
     * 13 - 15: float sun emittance (color * intensity ^ gamma), used for direct light
     * 16: float luminosity pdf (1 / luminosity)
     */
    @Override
    public IntArrayList pack() {
        IntArrayList out = new IntArrayList(17);
        out.add(flags);
        out.add((int) (texture >>> 32));
        out.add((int) texture);
        out.add(Float.floatToIntBits(altitude));
        out.add(Float.floatToIntBits(azimuth));
        out.add(Float.floatToIntBits(luminosity));
        out.add(Float.floatToIntBits(radius));
        for (float c : color) out.add(Float.floatToIntBits(c));
        for (float b : apparentTextureBrightness) out.add(Float.floatToIntBits(b));
        for (float e : emittance) out.add(Float.floatToIntBits(e));
        out.add(Float.floatToIntBits(luminosityPdf));
        return out;
    }
}
