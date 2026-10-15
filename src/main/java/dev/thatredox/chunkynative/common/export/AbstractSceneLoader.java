package dev.thatredox.chunkynative.common.export;

import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedBvhNode;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import dev.thatredox.chunkynative.common.export.models.PackedTriangleModel;
import dev.thatredox.chunkynative.common.export.primitives.PackedBlock;
import dev.thatredox.chunkynative.common.export.primitives.PackedMaterial;
import dev.thatredox.chunkynative.common.export.primitives.PackedSun;
import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.util.Reflection;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.SceneEntities;
import se.llbit.chunky.renderer.scene.sky.Sun;
import se.llbit.log.Log;
import se.llbit.math.Octree;
import se.llbit.math.PackedOctree;
import se.llbit.math.bvh.BVH;
import se.llbit.math.bvh.BinaryBVH;
import se.llbit.math.primitive.TexturedTriangle;

import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.util.Arrays;

public abstract class AbstractSceneLoader {
    protected int modCount = 0;
    protected WeakReference<BVH> prevWorldBvh = new WeakReference<>(null, null);
    protected WeakReference<BVH> prevActorBvh = new WeakReference<>(null, null);
    protected WeakReference<Octree.OctreeImplementation> prevOctree = new WeakReference<>(null, null);
    protected WeakReference<Octree.OctreeImplementation> prevWaterOctree = new WeakReference<>(null, null);

    protected AbstractTextureLoader texturePalette = null;
    protected ResourcePalette<PackedBlock> blockPalette = null;
    protected CachedResourcePalette<PackedMaterial> materialPalette = null;
    protected ResourcePalette<PackedAabbModel> aabbPalette = null;
    protected ResourcePalette<PackedQuadModel> quadPalette = null;
    protected ResourcePalette<PackedTriangleModel> trigPalette = null;
    protected int[] worldBvh = null;
    protected int[] actorBvh = null;
    protected PackedSun packedSun = null;

    public boolean ensureLoad(Scene scene) {
        return this.ensureLoad(scene, false);
    }

    protected boolean ensureLoad(Scene scene, boolean force) {
        if (force ||
                this.texturePalette == null || this.blockPalette == null || this.materialPalette == null ||
                this.aabbPalette == null || this.quadPalette == null || this.trigPalette == null ||
                this.prevOctree.get() != scene.getWorldOctree().getImplementation() ||
                this.prevWaterOctree.get() != scene.getWaterOctree().getImplementation()) {
            this.modCount = -1;
            return this.load(0, ResetReason.SCENE_LOADED, scene);
        }
        return true;
    }

    /**
     * @return True if successfully loaded. False if loading failed.
     */
    public boolean load(int modCount, ResetReason resetReason, Scene scene) {
        // Already up to date
        if (this.modCount == modCount) {
            return true;
        }

        if (resetReason == ResetReason.NONE || resetReason == ResetReason.MODE_CHANGE) {
            this.modCount = modCount;
            return true;
        }

        AbstractTextureLoader texturePalette = this.createTextureLoader();
        ResourcePalette<PackedBlock> blockPalette = this.createBlockPalette();
        CachedResourcePalette<PackedMaterial> materialPalette = new CachedResourcePalette<>(this.createMaterialPalette());
        ResourcePalette<PackedAabbModel> aabbPalette = this.createAabbModelPalette();
        ResourcePalette<PackedQuadModel> quadPalette = this.createQuadModelPalette();
        ResourcePalette<PackedTriangleModel> trigPalette = this.createTriangleModelPalette();

        SceneEntities entities = Reflection.getFieldValue(scene, "entities", SceneEntities.class);
        BVH worldBvh = Reflection.getFieldValue(entities, "bvh", BVH.class);
        BVH actorBvh = Reflection.getFieldValue(entities, "actorBvh", BVH.class);

        boolean needTextureLoad = resetReason == ResetReason.SCENE_LOADED ||
                resetReason == ResetReason.MATERIALS_CHANGED ||
                prevWorldBvh.get() != worldBvh ||
                prevActorBvh.get() != actorBvh;

        if (needTextureLoad) {
            if (!(worldBvh instanceof BinaryBVH || worldBvh == BVH.EMPTY)) {
                Log.error("BVH implementation must extend BinaryBVH");
                return false;
            }
            if (!(actorBvh instanceof BinaryBVH || actorBvh == BVH.EMPTY)) {
                Log.error("BVH implementation must extend BinaryBVH");
                return false;
            }
            prevWorldBvh = new WeakReference<>(worldBvh, null);
            prevActorBvh = new WeakReference<>(actorBvh, null);
        }

        // Preload textures
        if (needTextureLoad) {
            scene.getPalette().getPalette().forEach(b -> PackedBlock.preloadTextures(b, texturePalette));
            if (worldBvh != BVH.EMPTY) preloadBvh((BinaryBVH) worldBvh, texturePalette);
            if (actorBvh != BVH.EMPTY) preloadBvh((BinaryBVH) actorBvh, texturePalette);
            texturePalette.get(Sun.texture);
            texturePalette.build();
        }

        int[] blockMapping = null;
        int[] packedWorldBvh;
        int[] packedActorBvh;

        if (needTextureLoad) {
            blockMapping = scene.getPalette().getPalette().stream()
                    .mapToInt(block ->
                            blockPalette.put(new PackedBlock(block, texturePalette, materialPalette, aabbPalette, quadPalette)))
                    .toArray();
            if (worldBvh != BVH.EMPTY) {
                packedWorldBvh = loadBvh((BinaryBVH) worldBvh, texturePalette, materialPalette, trigPalette);
            } else {
                packedWorldBvh = PackedBvhNode.EMPTY_NODE.node;
            }
            if (actorBvh != BVH.EMPTY) {
                packedActorBvh = loadBvh((BinaryBVH) actorBvh, texturePalette, materialPalette, trigPalette);
            } else {
                packedActorBvh = PackedBvhNode.EMPTY_NODE.node;
            }
            packedSun = new PackedSun(scene.sun(), texturePalette);

            if (this.texturePalette != null) this.texturePalette.release();
            if (this.blockPalette != null) this.blockPalette.release();
            if (this.materialPalette != null) this.materialPalette.release();
            if (this.aabbPalette != null) this.aabbPalette.release();
            if (this.quadPalette != null) this.quadPalette.release();
            if (this.trigPalette != null) this.trigPalette.release();

            this.texturePalette = texturePalette;
            this.blockPalette = blockPalette;
            this.materialPalette = materialPalette;
            this.aabbPalette = aabbPalette;
            this.quadPalette = quadPalette;
            this.trigPalette = trigPalette;
            this.worldBvh = packedWorldBvh;
            this.actorBvh = packedActorBvh;
        }

        // Need to reload octrees
        Octree.OctreeImplementation impl = scene.getWorldOctree().getImplementation();
        Octree.OctreeImplementation waterImpl = scene.getWaterOctree().getImplementation();
        if (resetReason == ResetReason.SCENE_LOADED || prevOctree.get() != impl || prevWaterOctree.get() != waterImpl) {
            if (!(impl instanceof PackedOctree) || !(waterImpl instanceof PackedOctree)) {
                Log.error("Octree implementation must be PACKED");
                return false;
            }
            assert blockMapping != null;
            prevOctree = new WeakReference<>(impl, null);
            prevWaterOctree = new WeakReference<>(waterImpl, null);
            PackedOctree octree = (PackedOctree) impl;
            PackedOctree waterOctree = (PackedOctree) waterImpl;
            if (!loadOctree(octree.treeData, usedLength(octree), impl.getDepth(), blockMapping, blockPalette))
                return false;
            if (!loadWaterOctree(waterOctree.treeData, usedLength(waterOctree), waterImpl.getDepth(), blockMapping, blockPalette))
                return false;
        }

        return true;
    }

    protected static void preloadBvh(BinaryBVH bvh, AbstractTextureLoader texturePalette) {
        Arrays.stream(bvh.packedPrimitives).flatMap(Arrays::stream).forEach(primitive -> {
            if (primitive instanceof TexturedTriangle) {
                texturePalette.get(((TexturedTriangle) primitive).material.texture);
            }
        });
    }

    protected static int[] loadBvh(BinaryBVH bvh,
                                  AbstractTextureLoader texturePalette,
                                  ResourcePalette<PackedMaterial> materialPalette,
                                  ResourcePalette<PackedTriangleModel> trigPalette) {
        int[] out = new int[bvh.packed.length];
        for (int i = 0; i < out.length; i += 7) {
            PackedBvhNode node = new PackedBvhNode(bvh.packed, i, bvh.packedPrimitives, texturePalette, materialPalette, trigPalette);
            System.arraycopy(node.pack().elements(), 0, out, i, 7);
        }
        return out;
    }

    /**
     * Upload an octree. Only the first {@code length} ints of {@code octree} are used: the array grows with
     * spare capacity (up to 1.5x).
     */
    protected abstract boolean loadOctree(int[] octree, int length, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette);

    protected abstract boolean loadWaterOctree(int[] octree, int length, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette);

    /** Ints of the octree's array in use (PackedOctree.size), or the whole array if that can't be read. */
    private static int usedLength(PackedOctree octree) {
        try {
            Field size = PackedOctree.class.getDeclaredField("size");
            size.setAccessible(true);
            return Math.min(size.getInt(octree), octree.treeData.length);
        } catch (ReflectiveOperationException e) {
            return octree.treeData.length;
        }
    }

    /**
     * The kernel's header for an octree with bounds (Octree_createWithContent): depth, then the
     * minimum and maximum corner of the blocks the kernel can hit, i.e. leaves that are not air or
     * ANY_TYPE once mapped with {@code blockMapping}. Empty octrees get an empty box.
     */
    protected static int[] octreeHeader(int[] octree, int length, int depth, int[] blockMapping) {
        int[] bounds = {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE};
        if (length > 0) addContentBounds(octree, 0, 0, 0, 0, depth, blockMapping, bounds);
        if (bounds[0] > bounds[3]) {
            return new int[] {depth, 0, 0, 0, 0, 0, 0};
        }
        return new int[] {depth, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]};
    }

    private static void addContentBounds(int[] octree, int index, int x, int y, int z, int level, int[] blockMapping, int[] bounds) {
        int value = octree[index];
        if (value > 0) {
            int half = 1 << (level - 1);
            for (int c = 0; c < 8; c++) {
                addContentBounds(octree, value + c, x + ((c >> 2) & 1) * half, y + ((c >> 1) & 1) * half, z + (c & 1) * half,
                        level - 1, blockMapping, bounds);
            }
            return;
        }
        int type = -value;
        int mapped = type < blockMapping.length ? blockMapping[type] : type;
        if (mapped == 0 || mapped == 0x7FFFFFFE) return;  // Air and ANY_TYPE, as the kernel treats them
        int size = 1 << level;
        bounds[0] = Math.min(bounds[0], x);
        bounds[1] = Math.min(bounds[1], y);
        bounds[2] = Math.min(bounds[2], z);
        bounds[3] = Math.max(bounds[3], x + size);
        bounds[4] = Math.max(bounds[4], y + size);
        bounds[5] = Math.max(bounds[5], z + size);
    }

    /** Ints of octree converted per upload step, so the converted octree is never copied whole in memory. */
    protected static final int OCTREE_UPLOAD_CHUNK = 1 << 22;

    /**
     * Convert {@code count} octree ints starting at {@code from} into {@code out}: Chunky block palette ids in
     * the leaves are replaced with pointers into the packed block palette.
     */
    protected static void mapOctree(int[] octree, int from, int[] out, int count, int[] blockMapping) {
        for (int i = 0; i < count; i++) {
            int value = octree[from + i];
            out[i] = value > 0 || -value >= blockMapping.length ? value : -blockMapping[-value];
        }
    }

    protected abstract AbstractTextureLoader createTextureLoader();
    protected abstract ResourcePalette<PackedBlock> createBlockPalette();
    protected abstract ResourcePalette<PackedMaterial> createMaterialPalette();
    protected abstract ResourcePalette<PackedAabbModel> createAabbModelPalette();
    protected abstract ResourcePalette<PackedQuadModel> createQuadModelPalette();
    protected abstract ResourcePalette<PackedTriangleModel> createTriangleModelPalette();
}
