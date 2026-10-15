#ifndef CHUNKYCL_HWRT_H
#define CHUNKYCL_HWRT_H

// Metal only (CHUNKYCL_HWRT): the world octree traced with Metal hardware ray tracing instead of the
// octree march. The acceleration structure holds one bounding box per group of blocks (a 2^n section
// with an occupancy bitmap, or one octree leaf). A small intersection function finds the first block
// along the ray that can be hit; the kernel then runs the same block test as Octree_octreeIntersect
// and, if the ray misses that block (a transparent pixel, a model that doesn't fill its block, the
// same block the ray is inside of), looks for the next one.

#include <metal_raytracing>
using namespace metal::raytracing;

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "octree.h"
#include "block.h"
#include "material.h"

// The first block that can be hit, found by HwRt_firstBlock.
typedef struct {
    int3 cell;
    int data;        // Packed block if known (octree leaves), 0 to look it up in the octree (sections)
    float leafExit;  // Distance at which the ray leaves the octree leaf (leaves only)
    int calls;       // Intersection function calls of the query (statistics)
} HwRtCandidate;

struct HwRtHit {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

// Box data written by the Metal bridge (buildBlockAccel in metal_bridge.m):
//   boxes:   6 floats per box (min, max) in octree coordinates
//   info:    per box (x, y, z, w): a section of 2^s blocks: section origin, w = its first bitmap word;
//            an octree leaf (of any size) of one block: w = -1 - packed block
//   bitmaps: one bit per block of a section (x << 2s | y << s | z), set if the block can be hit
//   params:  section shift s (0: leaves only), octree depth
[[intersection(bounding_box)]]
HwRtHit HwRt_firstBlock(float3 origin [[origin]],
                        float3 direction [[direction]],
                        float minDistance [[min_distance]],
                        float maxDistance [[max_distance]],
                        uint prim [[primitive_id]],
                        ray_data HwRtCandidate& candidate [[payload]],
                        device const float* boxes [[buffer(0)]],
                        device const int4* info [[buffer(1)]],
                        device const uint* bitmaps [[buffer(2)]],
                        device const int* params [[buffer(3)]]) {
    HwRtHit result;
    result.accept = false;
    result.distance = 0;
    candidate.calls++;

    float3 lo = float3(boxes[prim * 6], boxes[prim * 6 + 1], boxes[prim * 6 + 2]);
    float3 hi = float3(boxes[prim * 6 + 3], boxes[prim * 6 + 4], boxes[prim * 6 + 5]);
    float3 invD = 1 / direction;
    float3 t0s = (lo - origin) * invD;
    float3 t1s = (hi - origin) * invD;
    float3 tmin = fmin(t0s, t1s);
    float3 tmax = fmax(t0s, t1s);
    float tn = fmax(fmax(tmin.x, tmin.y), tmin.z);
    float tf = fmin(fmin(tmax.x, tmax.y), tmax.z);
    int axis = tn == tmin.x ? 0 : (tn == tmin.y ? 1 : 2);
    float t = fmax(tn, minDistance);
    if (t > tf || t >= maxDistance) return result;

    // First block: on the entry face, or the block at the start of the ray (a point on a block
    // boundary belongs to the block the ray moves into, as in Octree_blockAt).
    int3 boxLo = int3(lo);
    int3 boxHi = int3(hi) - 1;
    int3 cell;
    if (tn >= minDistance) {
        cell = int3(floor(origin + direction * t));
        cell[axis] = direction[axis] > 0 ? boxLo[axis] : boxHi[axis];
    } else {
        cell = Octree_blockAt(origin + direction * t, direction);
    }
    cell = clamp(cell, boxLo, boxHi);

    int4 inf = info[prim];
    if (inf.w < 0) {
        candidate.cell = cell;
        candidate.data = -1 - inf.w;
        candidate.leafExit = tf;
        result.accept = true;
        result.distance = t;
        return result;
    }

    int shift = params[0];
    int3 step = int3(direction.x > 0 ? 1 : -1, direction.y > 0 ? 1 : -1, direction.z > 0 ? 1 : -1);
    float3 next;
    for (int a = 0; a < 3; a++) {
        next[a] = direction[a] == 0 ? INFINITY : (float(cell[a] + (direction[a] > 0 ? 1 : 0)) - origin[a]) * invD[a];
    }
    float3 tDelta = fabs(invD);
    for (int k = 0; k < 3 << shift; k++) {
        int3 local = cell - inf.xyz;
        int bit = (local.x << (2 * shift)) | (local.y << shift) | local.z;
        if ((bitmaps[inf.w + (bit >> 5)] >> (bit & 31)) & 1) {
            candidate.cell = cell;
            candidate.data = 0;
            candidate.leafExit = 0;
            result.accept = true;
            result.distance = t;
            return result;
        }
        if (next.x < next.y && next.x < next.z) {
            t = next.x; cell.x += step.x; next.x += tDelta.x;
        } else if (next.y < next.z) {
            t = next.y; cell.y += step.y; next.y += tDelta.y;
        } else {
            t = next.z; cell.z += step.z; next.z += tDelta.z;
        }
        if (t >= maxDistance || any(cell < boxLo) || any(cell > boxHi)) break;
    }
    return result;
}

// Where a trace of the world octree stands between hardware queries: the distance to continue from,
// the block the ray is inside of (as Octree_octreeIntersect tracks it) and the last block tested.
typedef struct {
    float start;
    int current;
    int3 lastCell;
    int queries;
} HwRtTrace;

HwRtTrace HwRt_begin(Ray ray) {
    HwRtTrace trace;
    trace.start = 0;
    trace.current = ray.material;
    trace.lastCell = int3(0x7FFFFFFF);
    trace.queries = 0;
    return trace;
}

// The first block along the ray from `start` that can be hit, within maxDistance (hardware query).
bool HwRt_query(primitive_acceleration_structure accel, intersection_function_table<> functions,
                Ray ray, float start, float maxDistance, __private HwRtCandidate* candidate, __private float* distance) {
    intersector<> isect;
    isect.assume_geometry_type(geometry_type::bounding_box);
    HwRtCandidate found;
    found.calls = 0;
    metal::raytracing::ray r(ray.origin, ray.direction, start, maxDistance);
    intersection_result<> result = isect.intersect(r, accel, functions, found);
    candidate->calls = found.calls;
    if (result.type != intersection_type::bounding_box) return false;
    *candidate = found;
    *distance = result.distance;
    return true;
}

// Tests the block found by HwRt_query at `distance`: true with the hit in record and sample, or
// false with trace->start advanced past the block (or the whole leaf) to look for the next one.
bool HwRt_resolve(__private HwRtTrace* trace, HwRtCandidate candidate, float distance,
                  Octree octree, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette,
                  Ray ray, __private IntersectionRecord* record, __private MaterialSample* sample) {
    int3 cell = candidate.cell;
    float3 ad = fabs(ray.origin);

    // Blocks in between that can't be hit (air) end the block the ray was in.
    float eps = fmax(OFFSET, fmax(fmax(ad.x, ad.y), ad.z) * 4.8e-7f + distance * 4.8e-7f);
    if (distance > trace->start + eps) trace->current = 0;

    int data = candidate.data != 0 ? candidate.data : Octree_get(&octree, cell.x, cell.y, cell.z);
    bool block = data != 0 && data != 0x7FFFFFFE;
    if (block && data != trace->current && !all(cell == trace->lastCell)) {
        IntersectionRecord tempRecord = *record;
        MaterialSample tempSample = *sample;
        if (BlockPalette_intersectNormalizedBlock(palette, atlas, materialPalette, data, cell, ray, &tempRecord, &tempSample)) {
            tempRecord.block = data;
            tempRecord.medium = trace->current;
            *record = tempRecord;
            *sample = tempSample;
            return true;
        }
    }
    trace->current = block && palette.blockPalette[data] == 1 ? data : 0;
    trace->lastCell = cell;

    // Continue after this block, or after the whole leaf when the ray is now inside its block.
    float3 lo = float3(cell);
    float3 invD = 1 / ray.direction;
    float3 tt = (select(lo, lo + 1, ray.direction > 0) - ray.origin) * invD;
    float exit = fmin(fmin(tt.x, tt.y), tt.z);
    if (candidate.data != 0 && trace->current == data) exit = fmax(exit, candidate.leafExit);
    trace->start = fmax(exit, trace->start) + eps;
    return false;
}

// Continues a trace with queries and tests until a block is hit, nothing more is found within
// record->distance, or 256 queries were made in all.
bool HwRt_continue(primitive_acceleration_structure accel, intersection_function_table<> functions,
                   __private HwRtTrace* trace, Octree octree, image2d_array_t atlas, BlockPalette palette,
                   MaterialPalette materialPalette, Ray ray, __private IntersectionRecord* record, __private MaterialSample* sample) {
    while (trace->queries < 256) {
        if (trace->start >= record->distance) return false;
        HwRtCandidate candidate;
        float distance;
        trace->queries++;
        if (!HwRt_query(accel, functions, ray, trace->start, record->distance, &candidate, &distance)) return false;
        if (HwRt_resolve(trace, candidate, distance, octree, atlas, palette, materialPalette, ray, record, sample)) return true;
    }
    return false;
}

// Closest block of the world octree along the ray within record->distance, like
// Octree_octreeIntersect (which this replaces), with the same tracking of the block the ray is in.
// *calls counts the hardware queries.
bool HwRt_intersect(primitive_acceleration_structure accel, intersection_function_table<> functions,
                    Octree octree, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette,
                    Ray ray, __private IntersectionRecord* record, __private MaterialSample* sample, __private int* calls) {
    HwRtTrace trace = HwRt_begin(ray);
    bool hit = HwRt_continue(accel, functions, &trace, octree, atlas, palette, materialPalette, ray, record, sample);
    *calls = trace.queries;
    return hit;
}

#endif
