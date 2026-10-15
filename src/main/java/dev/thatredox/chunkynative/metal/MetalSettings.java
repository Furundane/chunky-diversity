package dev.thatredox.chunkynative.metal;

import se.llbit.chunky.PersistentSettings;

/**
 * Saved settings of the Metal renderer (Metal tab). A system property that is set overrides the
 * saved value, e.g. for tests: -Dchunkymetal.wavefront=false.
 */
public final class MetalSettings {
    private static final String WAVEFRONT = "metalWavefront";
    private static final String HARDWARE_WATER = "metalHardwareWater";

    private MetalSettings() {}

    /**
     * Whether the Metal renderer runs the wavefront render loop (small kernels per bounce over the
     * live paths) instead of the single render kernel. On by default: same images, faster.
     */
    public static boolean wavefront() {
        return flag("chunkymetal.wavefront", WAVEFRONT, true);
    }

    public static void setWavefront(boolean value) {
        set(WAVEFRONT, value);
    }

    /**
     * Whether the wavefront renderer also traces the water octree with ray tracing hardware (when
     * block tracing uses it). Off by default: no faster on the test worlds.
     */
    public static boolean hardwareWater() {
        return flag("chunkymetal.hwWater", HARDWARE_WATER, false);
    }

    public static void setHardwareWater(boolean value) {
        set(HARDWARE_WATER, value);
    }

    private static boolean flag(String property, String setting, boolean defaultValue) {
        String value = System.getProperty(property, "");
        if (!value.isEmpty()) return Boolean.parseBoolean(value);
        return PersistentSettings.settings.getBool(setting, defaultValue);
    }

    private static void set(String setting, boolean value) {
        PersistentSettings.settings.setBool(setting, value);
        PersistentSettings.save();
    }
}
