#ifndef CHUNKYCLPLUGIN_OCTREE_H
#define CHUNKYCLPLUGIN_OCTREE_H

#include "../opencl.h"
#include "rt.h"
#include "constants.h"
#include "primitives.h"
#include "block.h"
#include "utils.h"

typedef struct {
    __global const int* treeData;
    AABB bounds;
    int depth;
} Octree;

Octree Octree_create(__global const int* treeData, int depth) {
    Octree octree;
    octree.treeData = treeData;
    octree.depth = depth;
    octree.bounds = AABB_new(0, 1<<depth, 0, 1<<depth, 0, 1<<depth);
    return octree;
}

int Octree_get(Octree* self, int x, int y, int z) {
    int3 bp = (int3) (x, y, z);

    // Check inbounds
    int3 lv = bp >> self->depth;
    if ((lv.x != 0) | (lv.y != 0) | (lv.z != 0))
        return 0;

    int level = self->depth;
    int data = self->treeData[0];
    while (data > 0) {
        level--;
        lv = 1 & (bp >> level);
        data = self->treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
    }
    return -data;
}

bool Octree_octreeIntersect(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, int drawDepth, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    float distMarch = 0;

    float3 invD = 1 / ray.direction;
    float3 offsetD = ray.direction * OFFSET;

    int depth = self.depth;
    int current = ray.material;  // Block the ray is inside of

    // Check if we are in bounds
    if (!AABB_inside(self.bounds, ray.origin)) {
        // Attempt to intersect with the octree
        float dist = AABB_quick_intersect(self.bounds, ray.origin, invD);
        if (isnan(dist) || dist < 0) {
            return false;
        } else {
            distMarch += dist + OFFSET;
        }
    }

    for (int i = 0; i < drawDepth; i++) {
        if (distMarch > record->distance) {
            // There's already been a closer intersection!
            return false;
        }

        float3 pos = ray.origin + ray.direction * distMarch;
        int3 bp = intFloorFloat3(pos + offsetD);

        // Check inbounds
        int3 lv = bp >> depth;
        if (lv.x != 0 || lv.y != 0 || lv.z != 0) {
            return false;
        }

        // Read the octree with depth
        int level = depth;
        int data = self.treeData[0];
        while (data > 0) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        // Like Chunky's Octree.enterBlock, a block next to the block the ray is in is only intersected if
        // it is a different block, so a ray that entered glass passes through the rest of that glass.
        if (data == 0 || data == 0x7FFFFFFE) {
            current = 0;  // Air
        } else if (data != current) {
            if (BlockPalette_intersectNormalizedBlock(palette, atlas, materialPalette, data, bp, ray, record, sample)) {
                record->block = data;
                record->medium = current;
                return true;
            }
            // Missed: inside a full block through a transparent pixel, or past a model that doesn't fill its block.
            current = palette.blockPalette[data] == 1 ? data : 0;
        }

        // Models that don't fill their block (AABB/quad models, light blocks, water) are only tested
        // against the block containing the ray, so step one block at a time through a merged leaf
        // of them instead of skipping the whole leaf (which drops e.g. lake surfaces).
        if (data != 0x7FFFFFFE && palette.blockPalette[data] >= 2) {
            level = 0;
            lv = bp;
        }

        // Exit the current leaf
        AABB box = AABB_new(lv.x << level, (lv.x + 1) << level,
                            lv.y << level, (lv.y + 1) << level,
                            lv.z << level, (lv.z + 1) << level);
        distMarch += AABB_exit(box, pos + offsetD, invD) + OFFSET;
    }
    return false;
}

bool _Octree_isWater(BlockPalette palette, int data) {
    return data != 0x7FFFFFFE && palette.blockPalette[data] == MODEL_WATER;
}

// Normal of the box face the ray leaves through, pointing along the ray.
float3 _Octree_exitNormal(AABB box, float3 origin, float3 invD) {
    float tx = (invD.x > 0 ? box.xmax - origin.x : box.xmin - origin.x) * invD.x;
    float ty = (invD.y > 0 ? box.ymax - origin.y : box.ymin - origin.y) * invD.y;
    float tz = (invD.z > 0 ? box.zmax - origin.z : box.zmin - origin.z) * invD.z;
    if (tx <= ty && tx <= tz) return (float3) (sign(invD.x), 0, 0);
    if (ty <= tz) return (float3) (0, sign(invD.y), 0);
    return (float3) (0, 0, sign(invD.z));
}

// Is the point inside water (used to find the starting medium of camera rays)?
bool Octree_isInsideWater(Octree self, BlockPalette palette, float3 point) {
    int3 bp = intFloorFloat3(point);
    int data = Octree_get(&self, bp.x, bp.y, bp.z);
    if (!_Octree_isWater(palette, data)) return false;
    WaterCorners corners = WaterCorners_new(palette.blockPalette[data + 2]);
    float3 inBlock = point - int3toFloat3(bp);
    return inBlock.y < WaterCorners_heightAt(corners, inBlock.x, inBlock.z);
}

// March a ray that is inside water through the water octree until it leaves the water, either
// through a water surface or into a non-water block. Mirrors Octree.exitWater in Chunky.
bool Octree_exitWater(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette,
                      int drawDepth, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    if (!AABB_inside(self.bounds, ray.origin)) {
        return false;
    }

    float distMarch = 0;
    float3 invD = 1 / ray.direction;
    float3 offsetD = ray.direction * OFFSET;
    int depth = self.depth;
    int waterMaterial = -1;
    float3 normal = (float3) (0, 1, 0);

    for (int i = 0; i < drawDepth; i++) {
        if (distMarch > record->distance) {
            return false;
        }

        float3 pos = ray.origin + ray.direction * distMarch;
        int3 bp = intFloorFloat3(pos + offsetD);
        int3 lv = bp >> depth;
        if (lv.x != 0 || lv.y != 0 || lv.z != 0) {
            return false;
        }

        int level = depth;
        int data = self.treeData[0];
        while (data > 0) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        float hitDistance = -1;
        if (!_Octree_isWater(palette, data)) {
            if (waterMaterial < 0) {
                // The ray started just outside the water volume; nothing to leave.
                return false;
            }
            hitDistance = distMarch;
        } else {
            waterMaterial = palette.blockPalette[data + 1];
            int waterData = palette.blockPalette[data + 2];
            if (!(waterData & WATER_FULL_BLOCK)) {
                // Partial block: the ray may leave through the water surface.
                WaterCorners corners = WaterCorners_new(waterData);
                float t = record->distance;
                float3 n;
                if (Water_intersectTop(corners, ray.origin - int3toFloat3(bp), ray.direction, EPS, &t, &n)) {
                    hitDistance = t;
                    normal = n;
                }
                level = 0;
                lv = bp;
            }
        }

        if (hitDistance >= 0) {
            if (hitDistance >= record->distance) {
                return false;
            }
            float3 hit = ray.origin + ray.direction * hitDistance;
            float2 uv = (float2) (hit.x - floor(hit.x), hit.z - floor(hit.z));
            Material material = Material_get(materialPalette, waterMaterial);
            if (!Material_sample(material, atlas, uv, sample)) {
                sample->color = (float4) (1, 1, 1, 1);
                sample->emittance = 0;
                sample->specular = 0;
                sample->metalness = 0;
                sample->roughness = 0;
            }
            record->distance = hitDistance;
            record->normal = normal;
            record->texCoord = uv;
            record->material = waterMaterial;
            record->flags = HIT_WATER;
            return true;
        }

        AABB box = AABB_new(lv.x << level, (lv.x + 1) << level,
                            lv.y << level, (lv.y + 1) << level,
                            lv.z << level, (lv.z + 1) << level);
        normal = _Octree_exitNormal(box, pos + offsetD, invD);
        distMarch += AABB_exit(box, pos + offsetD, invD) + OFFSET;
    }
    return false;
}

#endif
