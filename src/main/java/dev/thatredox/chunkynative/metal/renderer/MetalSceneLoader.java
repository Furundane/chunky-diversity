package dev.thatredox.chunkynative.metal.renderer;

import dev.thatredox.chunkynative.common.export.AbstractSceneLoader;
import dev.thatredox.chunkynative.common.export.Packer;
import dev.thatredox.chunkynative.common.export.ResourcePalette;
import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import dev.thatredox.chunkynative.common.export.models.PackedTriangleModel;
import dev.thatredox.chunkynative.common.export.primitives.PackedBlock;
import dev.thatredox.chunkynative.common.export.primitives.PackedMaterial;
import dev.thatredox.chunkynative.common.export.primitives.PackedSun;
import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.scene.SkyData;
import dev.thatredox.chunkynative.common.state.SkyState;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import dev.thatredox.chunkynative.metal.util.MetalTexture;
import dev.thatredox.chunkynative.util.FunctionCache;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.Arrays;

/** Uploads a scene to Metal buffers and textures. Metal version of ClSceneLoader. */
public class MetalSceneLoader extends AbstractSceneLoader {
    private final long context;
    protected final FunctionCache<int[], MetalBuffer> worldBvhBuffer;
    protected final FunctionCache<int[], MetalBuffer> actorBvhBuffer;
    protected final FunctionCache<PackedSun, MetalBuffer> sunBuffer;
    protected MetalTexture skyTexture = null;
    protected MetalBuffer skyIntensity = null;
    protected SkyState skyState = null;

    protected MetalBuffer octreeData = null;
    protected MetalBuffer octreeDepth = null;
    protected MetalBuffer waterOctreeData = null;
    protected MetalBuffer waterOctreeDepth = null;

    public MetalSceneLoader(long context) {
        this.context = context;
        this.worldBvhBuffer = new FunctionCache<>(i -> MetalBuffer.ofInts(context, i), MetalBuffer::close, null);
        this.actorBvhBuffer = new FunctionCache<>(i -> MetalBuffer.ofInts(context, i), MetalBuffer::close, null);
        this.sunBuffer = new FunctionCache<>(i -> MetalBuffer.ofInts(context, i.pack()), MetalBuffer::close, null);
    }

    @Override
    public boolean ensureLoad(Scene scene) {
        return this.ensureLoad(scene, skyTexture == null);
    }

    @Override
    public boolean load(int modCount, ResetReason resetReason, Scene scene) {
        boolean loadSuccess = super.load(modCount, resetReason, scene);
        if (this.modCount != modCount) {
            SkyState newSky = new SkyState(scene.sky(), scene.sun());
            if (!newSky.equals(skyState)) {
                if (skyTexture != null) skyTexture.close();
                if (skyIntensity != null) skyIntensity.close();
                SkyData sky = new SkyData(scene);
                skyTexture = MetalTexture.create2D(context, sky.resolution, sky.resolution, sky.texture);
                skyIntensity = MetalBuffer.ofFloats(context, new float[] {sky.intensity});
                skyState = newSky;
                packedSun = new PackedSun(scene.sun(), getTexturePalette());
            }
        }
        return loadSuccess;
    }

    @Override
    protected boolean loadOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (octreeData != null) octreeData.close();
        if (octreeDepth != null) octreeDepth.close();
        octreeData = MetalBuffer.ofInts(context, mapOctree(octree, blockMapping));
        octreeDepth = MetalBuffer.ofInt(context, depth);
        return true;
    }

    @Override
    protected boolean loadWaterOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (waterOctreeData != null) waterOctreeData.close();
        if (waterOctreeDepth != null) waterOctreeDepth.close();
        waterOctreeData = MetalBuffer.ofInts(context, mapOctree(octree, blockMapping));
        waterOctreeDepth = MetalBuffer.ofInt(context, depth);
        return true;
    }

    /** Replace Chunky block palette ids in the leaves with pointers into the packed block palette. */
    private static int[] mapOctree(int[] octree, int[] blockMapping) {
        return Arrays.stream(octree)
                .map(i -> i > 0 || -i >= blockMapping.length ? i : -blockMapping[-i])
                .toArray();
    }

    @Override
    protected AbstractTextureLoader createTextureLoader() {
        return new MetalTextureLoader(context);
    }

    @Override
    protected ResourcePalette<PackedBlock> createBlockPalette() {
        return new MetalPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedMaterial> createMaterialPalette() {
        return new MetalPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedAabbModel> createAabbModelPalette() {
        return new MetalPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedQuadModel> createQuadModelPalette() {
        return new MetalPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedTriangleModel> createTriangleModelPalette() {
        return new MetalPackedResourcePalette<>(context);
    }

    public MetalBuffer getOctreeData() {
        return octreeData;
    }

    public MetalBuffer getOctreeDepth() {
        return octreeDepth;
    }

    public MetalBuffer getWaterOctreeData() {
        return waterOctreeData;
    }

    public MetalBuffer getWaterOctreeDepth() {
        return waterOctreeDepth;
    }

    public MetalTextureLoader getTexturePalette() {
        return (MetalTextureLoader) texturePalette;
    }

    @SuppressWarnings("unchecked")
    private static <T extends Packer> MetalBuffer buffer(ResourcePalette<T> palette) {
        return ((MetalPackedResourcePalette<T>) palette).get();
    }

    public MetalBuffer getBlockPalette() {
        return buffer(blockPalette);
    }

    public MetalBuffer getMaterialPalette() {
        return buffer(materialPalette.palette);
    }

    public MetalBuffer getAabbPalette() {
        return buffer(aabbPalette);
    }

    public MetalBuffer getQuadPalette() {
        return buffer(quadPalette);
    }

    public MetalBuffer getTrigPalette() {
        return buffer(trigPalette);
    }

    public MetalBuffer getWorldBvh() {
        return worldBvhBuffer.apply(this.worldBvh);
    }

    public MetalBuffer getActorBvh() {
        return actorBvhBuffer.apply(this.actorBvh);
    }

    public MetalTexture getSkyTexture() {
        return skyTexture;
    }

    public MetalBuffer getSkyIntensity() {
        return skyIntensity;
    }

    public MetalBuffer getSun() {
        return sunBuffer.apply(packedSun);
    }
}
