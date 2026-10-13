package dev.thatredox.chunkynative.common.export.texture;

import dev.thatredox.chunkynative.opencl.renderer.export.textureexporter.TextureExporter;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import se.llbit.chunky.resources.Texture;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Packs textures into an array of RGBA8 atlas layers of {@link #ATLAS_SIZE} pixels square, in
 * 16 pixel cells. Subclasses upload the packed textures to a device.
 */
public abstract class AtlasTextureLoader extends AbstractTextureLoader {
    public static final int ATLAS_SIZE = 8192;

    /**
     * Create the atlas with the given number of layers and write every texture at its location
     * ({@link AtlasTexture#getX()} etc., in 16 pixel cells).
     */
    protected abstract void upload(int layers, List<AtlasTexture> textures);

    @Override
    protected void buildTextures(Object2ObjectMap<Texture, TextureRecord> textures) {
        List<AtlasTexture> texs = textures.entrySet().stream()
                .map(entry -> new AtlasTexture(entry.getKey(), entry.getValue()))
                .sorted().collect(Collectors.toList());

        ArrayList<boolean[][]> layers = new ArrayList<>();
        layers.add(new boolean[256][256]);
        for (AtlasTexture tex : texs) {
            if (!insertTex(layers, tex)) {
                layers.add(new boolean[256][256]);
                insertTex(layers, tex);
            }
        }

        upload(layers.size(), texs);

        texs.forEach(AtlasTexture::commit);
    }

    private static boolean insertTex(ArrayList<boolean[][]> layers, AtlasTexture tex) {
        int l = 0;
        int blockWidth = Math.max(1, tex.getWidth() / 16);
        int blockHeight = Math.max(1, tex.getHeight() / 16);
        for (boolean[][] layer : layers) {
            for (int x = 0; x < 256; x++) {
                for (int y = 0; y < 256; y++) {
                    if (insertAt(x, y, blockWidth, blockHeight, layer)) {
                        tex.setLocation(x, y, l);
                        return true;
                    }
                }
            }
            l++;
        }
        return false;
    }

    private static boolean insertAt(int x, int y, int width, int height, boolean[][] layer) {
        if (y + height > layer.length || x + width > layer[0].length) {
            return false;
        }

        if (y < 0 || x < 0) {
            return false;
        }

        for (int line = y; line < y + height; line++) {
            for (int pixel = x; pixel < x + width; pixel++) {
                if (layer[line][pixel]) {
                    return false;
                }
            }
        }

        for (int line = y; line < y + height; line++) {
            for (int pixel = x; pixel < x + width; pixel++) {
                layer[line][pixel] = true;
            }
        }

        return true;
    }

    protected static class AtlasTexture implements Comparable<AtlasTexture> {
        public final TextureExporter exporter;
        public final TextureRecord record;
        public final int size;
        public int location = 0xFFFFFFFF;

        protected AtlasTexture(Texture tex, TextureRecord record) {
            this.exporter = TextureExporter.getExporter(tex);
            this.record = record;
            this.size = (exporter.getWidth() << 16) | exporter.getHeight();
        }

        public void commit() {
            this.record.set(((long) size << 32) | location);
        }

        public void setLocation(int x, int y, int d) {
            this.location = (x << 22) | (y << 13) | d;
        }

        public int getWidth() {
            return (size >>> 16) & 0xFFFF;
        }

        public int getHeight() {
            return size & 0xFFFF;
        }

        public int getX() {
            return (location >>> 22) & 0x1FF;
        }

        public int getY() {
            return (location >>> 13) & 0x1FF;
        }

        public int getD() {
            return location & 0x1FFF;
        }

        public byte[] getTexture() {
            return exporter.getTexture();
        }

        @Override
        public int compareTo(AtlasTexture o) {
            return o.size - this.size;
        }

        @Override
        public int hashCode() {
            return exporter.textureHashCode();
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) return false;
            if (!(o instanceof AtlasTexture)) return false;
            AtlasTexture other = (AtlasTexture) o;
            if (this.size != other.size) return false;
            return this.exporter.equals(other.exporter);
        }
    }
}
