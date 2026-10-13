#ifndef CHUNKYCLPLUGIN_WAVEFRONT_H
#define CHUNKYCLPLUGIN_WAVEFRONT_H

#include "../opencl.h"
#include "compat.h"

#define RAY_INDIRECT 0b001
#define RAY_PREVIEW  0b010
#define RAY_WATER    0b100  // The ray is currently travelling through water
#define RAY_DIFFUSE  0b1000 // The last bounce was a diffuse reflection

#define HIT_WATER    0b1    // The intersection is a water surface (entering or leaving water)

typedef struct {
    float3 origin;
    float3 direction;
    int material;  // Packed block the ray is inside of (0 = air). Neighbors of the same block are not intersected.
    int flags;
} Ray;

typedef struct {
    float distance;
    int material;
    int flags;
    int block;   // Packed block that was hit (0 if not an octree block)
    int medium;  // Packed block the ray was inside of when it hit

    float3 normal;
    float2 texCoord;
} IntersectionRecord;

IntersectionRecord IntersectionRecord_new() {
    IntersectionRecord record;
    record.distance = HUGE_VALF;
    record.material = 0;
    record.flags = 0;
    record.block = 0;
    record.medium = 0;
    record.normal = make_float3(0, 1, 0);
    return record;
}

#endif
