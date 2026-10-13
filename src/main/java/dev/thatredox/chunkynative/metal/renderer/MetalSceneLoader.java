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
import dev.thatredox.chunkynative.metal.BlockAcceleration;
import dev.thatredox.chunkynative.metal.Metal;
import dev.thatredox.chunkynative.metal.util.MetalBuffer;
import dev.thatredox.chunkynative.util.FunctionCache;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Log;

import java.util.function.Consumer;


/** Uploads a scene to Metal buffers and textures. Metal version of ClSceneLoader. */
public class MetalSceneLoader extends AbstractSceneLoader {
    private final long context;
    protected final FunctionCache<int[], MetalBuffer> worldBvhBuffer;
    protected final FunctionCache<int[], MetalBuffer> actorBvhBuffer;
    protected final FunctionCache<PackedSun, MetalBuffer> sunBuffer;
    protected MetalBuffer skyTexture = null;
    protected MetalBuffer skyLight = null;
    protected SkyState skyState = null;

    protected MetalBuffer octreeData = null;
    protected MetalBuffer octreeDepth = null;
    protected MetalBuffer waterOctreeData = null;
    protected MetalBuffer waterOctreeDepth = null;
    private int worldDepth;

    // Block acceleration structure (hardware ray tracing) and what it was built from.
    private long blockAccel = 0;
    private BlockAcceleration blockAccelMode = null;
    private MetalBuffer blockAccelOctree = null;
    private ResourcePalette<PackedBlock> blockAccelPalette = null;
    private static volatile String accelerationStatus = "";
    private static volatile Consumer<String> statusListener = null;

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
                if (skyLight != null) skyLight.close();
                SkyData sky = new SkyData(scene);
                skyTexture = MetalBuffer.ofFloats(context, sky.texture);
                skyLight = MetalBuffer.ofFloats(context, sky.skyLight);
                skyState = newSky;
                packedSun = new PackedSun(scene.sun(), getTexturePalette());
            }
        }
        return loadSuccess;
    }

    @Override
    protected boolean loadOctree(int[] octree, int length, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        releaseBlockAcceleration();
        if (octreeData != null) octreeData.close();
        if (octreeDepth != null) octreeDepth.close();
        octreeData = uploadOctree(octree, length, blockMapping);
        octreeDepth = MetalBuffer.ofInt(context, depth);
        worldDepth = depth;
        return true;
    }

    @Override
    protected boolean loadWaterOctree(int[] octree, int length, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (waterOctreeData != null) waterOctreeData.close();
        if (waterOctreeDepth != null) waterOctreeDepth.close();
        waterOctreeData = uploadOctree(octree, length, blockMapping);
        waterOctreeDepth = MetalBuffer.ofInt(context, depth);
        return true;
    }

    private MetalBuffer uploadOctree(int[] octree, int length, int[] blockMapping) {
        MetalBuffer buffer = new MetalBuffer(context, 4L * length);
        int[] chunk = new int[Math.max(1, Math.min(length, OCTREE_UPLOAD_CHUNK))];
        for (int start = 0; start < length; start += chunk.length) {
            int count = Math.min(chunk.length, length - start);
            mapOctree(octree, start, chunk, count, blockMapping);
            buffer.writeInts(start, chunk, 0, count);
        }
        return buffer;
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

    /**
     * The block acceleration structure for the current {@link BlockAcceleration} setting, built when
     * the setting, the world octree or the block palette changed; 0 to use the octree march (also
     * when the structure can't be built, e.g. without enough memory). Callers that keep the handle
     * beyond the current render take their own reference ({@link Metal#retain}).
     */
    public synchronized long blockAcceleration() {
        BlockAcceleration mode = BlockAcceleration.current();
        if (mode == blockAccelMode && octreeData == blockAccelOctree && blockPalette == blockAccelPalette) {
            return blockAccel;
        }
        releaseBlockAcceleration();
        blockAccelMode = mode;
        blockAccelOctree = octreeData;
        blockAccelPalette = blockPalette;
        if (mode == BlockAcceleration.OFF || octreeData == null) {
            setStatus("Using the octree");
            return 0;
        }
        try {
            blockAccel = Metal.buildBlockAccel(context, octreeData.handle, worldDepth, getBlockPalette().handle, mode.sectionShift, 0);
            long[] info = Metal.blockAccelInfo(blockAccel);
            setStatus(String.format("%,d boxes, %.0f MB, built in %.1f s", info[0], (info[1] + info[2]) / 1e6, info[3] / 1e9));
            Log.infof("ChunkyCL Metal: %s: %s", mode.label, accelerationStatus);
        } catch (RuntimeException e) {
            blockAccel = 0;
            setStatus("Using the octree: " + e.getMessage());
            Log.warn("ChunkyCL Metal: could not use ray tracing hardware: " + e.getMessage());
        }
        return blockAccel;
    }

    private synchronized void releaseBlockAcceleration() {
        Metal.release(blockAccel);
        blockAccel = 0;
        blockAccelMode = null;
        blockAccelOctree = null;
        blockAccelPalette = null;
    }

    private static void setStatus(String status) {
        accelerationStatus = status;
        Consumer<String> listener = statusListener;
        if (listener != null) listener.accept(status);
    }

    /** What the block acceleration setting currently does, for the UI. */
    public static String accelerationStatus() {
        return accelerationStatus;
    }

    public static void setStatusListener(Consumer<String> listener) {
        statusListener = listener;
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

    public MetalBuffer getSkyTexture() {
        return skyTexture;
    }

    public MetalBuffer getSkyLight() {
        return skyLight;
    }

    public MetalBuffer getSun() {
        return sunBuffer.apply(packedSun);
    }
}
