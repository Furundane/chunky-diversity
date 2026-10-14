package dev.thatredox.chunkynative.metal;

import se.llbit.chunky.PersistentSettings;

/**
 * How the Metal renderer finds the blocks a ray hits: the octree march shared with OpenCL, or Metal
 * hardware ray tracing over boxes of blocks (kernel/include/hwrt.h). Larger boxes need less memory,
 * smaller ones are faster.
 */
public enum BlockAcceleration {
    OFF("Octree (no ray tracing hardware)", -1),
    SECTIONS_8("Ray tracing, 8x8x8 sections (least memory)", 3),
    SECTIONS_4("Ray tracing, 4x4x4 sections (balanced)", 2),
    LEAVES("Ray tracing, octree leaves (fastest, most memory)", 0);

    private static final String SETTING = "metalBlockAcceleration";
    /** Overrides the saved setting, e.g. for tests: -Dchunkymetal.blockAcceleration=SECTIONS_4 */
    private static final String PROPERTY = "chunkymetal.blockAcceleration";

    public final String label;
    /** Section size 2^shift, 0 for octree leaves. */
    public final int sectionShift;

    BlockAcceleration(String label, int sectionShift) {
        this.label = label;
        this.sectionShift = sectionShift;
    }

    public static BlockAcceleration current() {
        String value = System.getProperty(PROPERTY, "");
        if (value.isEmpty()) {
            value = PersistentSettings.settings.getString(SETTING, OFF.name());
        }
        try {
            return valueOf(value);
        } catch (IllegalArgumentException e) {
            return OFF;
        }
    }

    public static void set(BlockAcceleration mode) {
        PersistentSettings.settings.setString(SETTING, mode.name());
        PersistentSettings.save();
    }

    @Override
    public String toString() {
        return label;
    }
}
