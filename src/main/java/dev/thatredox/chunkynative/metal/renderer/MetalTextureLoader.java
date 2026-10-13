package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.common.export.texture.AtlasTextureLoader;
import dev.thatredox.chunkynative.metal.util.MetalTexture;

import java.util.List;

/** The texture atlas as a Metal 2D texture array. */
public class MetalTextureLoader extends AtlasTextureLoader {
    private MetalTexture atlas;
    private final long context;

    public MetalTextureLoader(long context) {
        this.context = context;
    }

    public MetalTexture getAtlas() {
        return atlas;
    }

    @Override
    protected void upload(int layers, List<AtlasTexture> textures) {
        atlas = MetalTexture.create2DArray(context, ATLAS_SIZE, ATLAS_SIZE, layers);
        for (AtlasTexture tex : textures) {
            atlas.write(tex.getX() * 16, tex.getY() * 16, tex.getD(), tex.getWidth(), tex.getHeight(), tex.getTexture());
        }
    }

    @Override
    public void release() {
        if (atlas != null) atlas.close();
    }
}
