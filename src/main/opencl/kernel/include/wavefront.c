// Wavefront version of the render kernel (rayTracer.c): the path tracing loop split into small
// kernels that run once per bounce over the live paths of a wave, so that the kernels which trace
// rays (wf_intersect, wf_shadow) are not burdened with the state of the whole path tracer.
//
// Per render pass and wave of paths: wf_generate (all paths, queued), then for each bounce
// wf_prepare, wf_intersect and wf_shade over the queued live paths, wf_prepareShadow and wf_shadow
// over the paths that requested sun light, then wf_finish (all paths). wf_shade queues the paths
// still alive for the next bounce; the prepare kernels turn queue lengths into indirect dispatch
// sizes on the GPU. Every path keeps its own random state and writes only its own pixel, so the
// order of the queues does not change the result: the paths do the same work as in the render
// kernel, in the same order and with the same random numbers.

#include "../opencl.h"
#include "compat.h"
#include "octree.h"
#include "rt.h"
#include "block.h"
#include "material.h"
#include "kernel.h"
#include "camera.h"
#include "bvh.h"
#include "sky.h"
#include "water.h"

// Path state, STATE_F floats in stateF and STATE_I ints in stateI per path of the wave, stored by
// field (field k of path p at k * wave[3] + p) so that neighboring paths read neighboring memory.
// Both are in the buffer `state`, the ints after the floats (see WF_BUFFERS):
//   path   (floats 0..11):  origin (3), direction (3), color (3), throughput (3)
//          (ints 0..3):     ray material, ray flags, random state, alive
//   hit    (floats 12..25): distance, normal (3), texCoord (2), sample color (4), emittance, specular,
//                           metalness, roughness
//          (ints 4..8):     hit, material, flags, block, medium
//   shadow (floats 26..34): origin (3), direction (3), contribution (3)  (sun light requested by wf_shade)
//          (ints 9..11):    requested, ray material, ray flags
//   trace  (floats 35..37): HwRtTrace start, candidate distance, candidate leafExit
//          (ints 12..21):   HwRtTrace current, lastCell (3), queries, candidate cell (3), candidate data,
//                           candidate found  (world octree traced with hardware ray tracing, see wf_trace)
#define STATE_F 38
#define STATE_I 22
#define TRACE_F 35
#define TRACE_I 12
#define HIT_F 12
#define HIT_I 4
#define SHADOW_F 26
#define SHADOW_I 9

// Queues: queues[k * wave[3] + i], k = 0 and 1: live paths (input and output alternate per bounce,
// wave[4] is the input), k = 2: paths that requested sun light, k = 3 and 4: paths whose world
// octree trace continues in the next round (wave[5] is the input). control: queue lengths
// (control[k] for k < 3, control[CONTROL_TRACE + k - 3] for the trace queues), the indirect
// dispatch arguments (threadgroups) for the path, shadow and trace queues, and statistics.
#define QUEUE_SHADOW 2
#define QUEUE_TRACE 3
#define CONTROL_ARGS 4
#define CONTROL_SHADOW_ARGS 8
#define CONTROL_STATS 12
#define CONTROL_TRACE 16
#define CONTROL_TRACE_ARGS 20
#define CONTROL_SIZE 32

// The path state and the control block and queues share two buffers (state, work), since Metal
// takes at most 31 buffer arguments. Every kernel starts with this.
#define WF_BUFFERS \
    __global float* stateF = state; \
    __global int* stateI = (__global int*) (state + STATE_F * wave[3]); \
    __global uint* control = work; \
    __global uint* queues = work + CONTROL_SIZE; \
    (void) stateF; (void) stateI; (void) control; (void) queues
// Threadgroup size of the indirect dispatches (WavefrontJob.GROUP_SIZE).
#define WF_GROUP 64

// Metal hardware ray tracing of the world octree (hwrt.h), compiled in with CHUNKYCL_HWRT.
#ifdef CHUNKYCL_HWRT
#define WF_HWRT_PARAMS primitive_acceleration_structure hwAccel, intersection_function_table<> hwFunctions,
#define WF_HWRT_SCENE(scene) scene.hwAccel = hwAccel; scene.hwFunctions = hwFunctions; scene.hwWorld = true
// The octree march in a kernel that should make no hardware calls.
#define WF_NO_HWRT(scene) scene.hwWorld = false
#else
#define WF_HWRT_PARAMS
#define WF_HWRT_SCENE(scene)
#define WF_NO_HWRT(scene)
#endif

// The water octree with hardware ray tracing too (CHUNKYCL_HWRT_WATER, see wf_traceWater).
#ifdef CHUNKYCL_HWRT_WATER
#define WF_HWRT_WATER_PARAMS primitive_acceleration_structure hwWaterAccel, intersection_function_table<> hwWaterFunctions,
#else
#define WF_HWRT_WATER_PARAMS
#endif

#ifdef CHUNKYCL_HWRT_STATS
#define WF_STATS_SCENE(scene) scene.stats = control + CONTROL_STATS
#else
#define WF_STATS_SCENE(scene)
#endif

// The parameters of every wavefront kernel: the render kernel's, then the path state and the wave
// ([0] first pixel, [1] number of paths, [2] bounce, [3] paths the state buffers hold, [4] input
// queue, [5] input trace queue, [6] last trace round, [7] leaves wf_resolve marches per round).
#define WF_PARAMS \
    __global const int* projectorType, \
    __global const float* cameraSettings, \
    __global const int* octreeDepth, \
    __global const int* octreeData, \
    WF_HWRT_PARAMS \
    __global const int* waterOctreeDepth, /* depth and bounds of the water, see Octree_createWithContent */ \
    __global const int* waterOctreeData, \
    WF_HWRT_WATER_PARAMS \
    __global const int* bPalette, \
    __global const int* quadModels, \
    __global const int* aabbModels, \
    __global const int* worldBvhData, \
    __global const int* actorBvhData, \
    __global const int* bvhTrigs, \
    image2d_array_t textureAtlas, \
    __global const int* matPalette, \
    __global const float* skyTexture, \
    __global const float* skyLight, \
    __global const int* sunData, \
    __global const float* waterSettings, \
    __global const int* atmosphere, \
    __global const int* randomSeed, \
    __global const int* bufferSpp, \
    __global const int* canvasConfig, \
    __global const int* rayDepth, \
    __global float* res, \
    __global float* state, \
    __global uint* work, \
    __global const int* wave

// The Scene of the render kernel. A macro, since the palettes keep pointers into the struct.
#define WF_SCENE(scene) \
    Scene scene; \
    scene.materialPalette = MaterialPalette_new(matPalette); \
    scene.octree = Octree_create(octreeData, *octreeDepth); \
    scene.waterOctree = Octree_createWithContent(waterOctreeData, waterOctreeDepth); \
    scene.water = WaterSettings_new(waterSettings); \
    scene.atmosphere = atmosphere; \
    scene.fogMode = atmosphere[0]; \
    scene.clouds = atmosphere[10] != 0; \
    scene.sunStrategy = (int) waterSettings[9]; \
    scene.emitterIntensity = waterSettings[10]; \
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs, &scene.materialPalette); \
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs, &scene.materialPalette); \
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, &scene.materialPalette); \
    scene.drawDepth = 256; \
    WF_HWRT_SCENE(scene); \
    WF_STATS_SCENE(scene)

// The state of one path: field k is f[k * stride] / in[k * stride].
typedef struct {
    __global float* f;
    __global int* in;
    int stride;
} PathState;

PathState WF_path(__global float* stateF, __global int* stateI, __global const int* wave, int p) {
    PathState s;
    s.f = stateF + p;
    s.in = stateI + p;
    s.stride = wave[3];
    return s;
}

float3 WF_get3(PathState s, int k) {
    return make_float3(s.f[k * s.stride], s.f[(k + 1) * s.stride], s.f[(k + 2) * s.stride]);
}

void WF_set3(PathState s, int k, float3 v) {
    s.f[k * s.stride] = v.x;
    s.f[(k + 1) * s.stride] = v.y;
    s.f[(k + 2) * s.stride] = v.z;
}

#define WF_F(s, k) ((s).f[(k) * (s).stride])
#define WF_I(s, k) ((s).in[(k) * (s).stride])

Ray WF_loadRay(PathState s) {
    Ray ray;
    ray.origin = WF_get3(s, 0);
    ray.direction = WF_get3(s, 3);
    ray.material = WF_I(s, 0);
    ray.flags = WF_I(s, 1);
    return ray;
}

void WF_storeRay(PathState s, Ray ray) {
    WF_set3(s, 0, ray.origin);
    WF_set3(s, 3, ray.direction);
    WF_I(s, 0) = ray.material;
    WF_I(s, 1) = ray.flags;
}

void WF_storeHit(PathState s, bool hit, IntersectionRecord record, MaterialSample sample) {
    WF_I(s, HIT_I) = hit ? 1 : 0;
    if (!hit) return;
    WF_F(s, HIT_F) = record.distance;
    WF_set3(s, HIT_F + 1, record.normal);
    WF_F(s, HIT_F + 4) = record.texCoord.x;
    WF_F(s, HIT_F + 5) = record.texCoord.y;
    WF_F(s, HIT_F + 6) = sample.color.x;
    WF_F(s, HIT_F + 7) = sample.color.y;
    WF_F(s, HIT_F + 8) = sample.color.z;
    WF_F(s, HIT_F + 9) = sample.color.w;
    WF_F(s, HIT_F + 10) = sample.emittance;
    WF_F(s, HIT_F + 11) = sample.specular;
    WF_F(s, HIT_F + 12) = sample.metalness;
    WF_F(s, HIT_F + 13) = sample.roughness;
    WF_I(s, HIT_I + 1) = record.material;
    WF_I(s, HIT_I + 2) = record.flags;
    WF_I(s, HIT_I + 3) = record.block;
    WF_I(s, HIT_I + 4) = record.medium;
}

void WF_loadHit(PathState s, __private IntersectionRecord* record, __private MaterialSample* sample) {
    record->distance = WF_F(s, HIT_F);
    record->normal = WF_get3(s, HIT_F + 1);
    record->texCoord = make_float2(WF_F(s, HIT_F + 4), WF_F(s, HIT_F + 5));
    sample->color = make_float4(WF_F(s, HIT_F + 6), WF_F(s, HIT_F + 7), WF_F(s, HIT_F + 8), WF_F(s, HIT_F + 9));
    sample->emittance = WF_F(s, HIT_F + 10);
    sample->specular = WF_F(s, HIT_F + 11);
    sample->metalness = WF_F(s, HIT_F + 12);
    sample->roughness = WF_F(s, HIT_F + 13);
    record->material = WF_I(s, HIT_I + 1);
    record->flags = WF_I(s, HIT_I + 2);
    record->block = WF_I(s, HIT_I + 3);
    record->medium = WF_I(s, HIT_I + 4);
}

// Scene_sunDirectLight without the shadow ray: the ray and the light it would add are left for
// wf_shadow, which multiplies the light by the attenuation along the ray.
void WF_requestSunLight(Scene self, Sun sun, Ray ray, IntersectionRecord record, Random random, float3 throughput, PathState s) {
    float3 normal = record.normal;
    if (dot(normal, ray.direction) > 0) normal = -normal;
    float3 sunDirection = Sun_sampleDirection(sun, random);
    float cosine = dot(sunDirection, normal);
    if (cosine <= 0) {
        return;
    }
    float mult = cosine * ((self.sunStrategy & SUN_LUMINOSITY) ? sun.luminosityPdf : 1);
    WF_set3(s, SHADOW_F, ray.origin + ray.direction * record.distance + normal * OFFSET);
    WF_set3(s, SHADOW_F + 3, sunDirection);
    WF_set3(s, SHADOW_F + 6, throughput * mult * sun.emittance);
    WF_I(s, SHADOW_I) = 1;
    WF_I(s, SHADOW_I + 1) = record.medium;
    WF_I(s, SHADOW_I + 2) = ray.flags & RAY_WATER;
}

__kernel void wf_generate(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
    int gid = wave[0] + p;
    PathState s = WF_path(stateF, stateI, wave, p);
    WF_SCENE(scene);

    unsigned int randomState = *randomSeed + gid;
    Random random = &randomState;
    Random_nextState(random);
    Ray ray = ray_to_camera(projectorType, cameraSettings, canvasConfig, gid, random);

    ray.material = 0;
    ray.flags = 0;
    if (Octree_isInsideWater(scene.waterOctree, scene.blockPalette, ray.origin)) {
        ray.flags |= RAY_WATER;
    }
    WF_storeRay(s, ray);
    WF_set3(s, 6, make_float3(0, 0, 0));
    WF_set3(s, 9, make_float3(1, 1, 1));
    WF_I(s, 2) = (int) randomState;
    WF_I(s, 3) = 1;
    WF_I(s, SHADOW_I) = 0;
    queues[p] = (uint) p;
    if (p == 0) control[0] = (uint) wave[1];
}

// Indirect dispatch size for the input queue of this bounce; empties the output and shadow queues.
// One thread.
__kernel void wf_prepare(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    if (GLOBAL_ID != 0) return;
    int in = wave[4];
    control[CONTROL_ARGS] = (control[in] + WF_GROUP - 1) / WF_GROUP;
    control[CONTROL_ARGS + 1] = 1;
    control[CONTROL_ARGS + 2] = 1;
    control[1 - in] = 0;
    control[QUEUE_SHADOW] = 0;
    control[CONTROL_TRACE] = 0;
}

// Indirect dispatch size for the shadow queue. One thread.
__kernel void wf_prepareShadow(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    if (GLOBAL_ID != 0) return;
    control[CONTROL_SHADOW_ARGS] = (control[QUEUE_SHADOW] + WF_GROUP - 1) / WF_GROUP;
    control[CONTROL_SHADOW_ARGS + 1] = 1;
    control[CONTROL_SHADOW_ARGS + 2] = 1;
}

// The path for thread i of a dispatch over queue k, or -1 past the end of the queue.
int WF_queued(__global const uint* queues, __global const uint* control, __global const int* wave, int k, int i) {
    if (i >= (int) control[k]) return -1;
    return (int) queues[k * wave[3] + i];
}

// Diagnostic (-Dchunkymetal.traceProbe=true): only the hardware query for the first block along each
// queued path's ray, the minimal kernel a split of wf_intersect would run. Its result is written to
// the hit distance, which wf_intersect overwrites.
__kernel void wf_traceProbe(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    intersector<> isect;
    isect.assume_geometry_type(geometry_type::bounding_box);
    HwRtCandidate candidate;
    metal::raytracing::ray r(WF_get3(s, 0), WF_get3(s, 3), 0.0f, INFINITY);
    intersection_result<> found = isect.intersect(r, hwAccel, hwFunctions, candidate);
    WF_F(s, HIT_F) = found.type == intersection_type::bounding_box ? found.distance + candidate.cell.x : -1.0f;
#endif
}

// The intersection of a bounce in parts, when the world octree is traced with hardware ray tracing:
// wf_begin, then rounds of wf_prepareTrace, wf_trace and wf_resolve, then wf_intersectRest. The
// hardware query (wf_trace) runs in a kernel of its own, which is much faster than in a kernel that
// also tests blocks; wf_resolve marches the octree for a few leaves from the block found and queues
// the paths that found nothing in them for the next round. The last round marches to the end.

// Russian roulette (as in wf_intersect), then starts the world octree trace of the surviving paths.
__kernel void wf_begin(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    if (wave[2] >= 3) {
        unsigned int randomState = (unsigned int) WF_I(s, 2);
        Random random = &randomState;
        float3 throughput = WF_get3(s, 9);
        float survival = fmin(1.0f, fmax(0.25f, fmax(throughput.x, fmax(throughput.y, throughput.z))));
        bool survives = Random_nextFloat(random) < survival;
        WF_I(s, 2) = (int) randomState;
        if (!survives) {
            WF_I(s, 3) = 0;
            return;
        }
        WF_set3(s, 9, throughput / survival);
    }
    WF_I(s, HIT_I) = 0;
    HwRtTrace trace = HwRt_begin(WF_loadRay(s));
    WF_F(s, TRACE_F) = trace.start;
    WF_I(s, TRACE_I) = trace.current;
    WF_I(s, TRACE_I + 1) = trace.lastCell.x;
    WF_I(s, TRACE_I + 2) = trace.lastCell.y;
    WF_I(s, TRACE_I + 3) = trace.lastCell.z;
    WF_I(s, TRACE_I + 4) = trace.queries;
    queues[QUEUE_TRACE * wave[3] + ATOMIC_ADD(control + CONTROL_TRACE, 1u)] = (uint) p;
#endif
}

// Indirect dispatch size for the input trace queue of this round; empties the output trace queue.
// One thread.
__kernel void wf_prepareTrace(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    if (GLOBAL_ID != 0) return;
    int in = wave[5];
    control[CONTROL_TRACE_ARGS] = (control[CONTROL_TRACE + in] + WF_GROUP - 1) / WF_GROUP;
    control[CONTROL_TRACE_ARGS + 1] = 1;
    control[CONTROL_TRACE_ARGS + 2] = 1;
    control[CONTROL_TRACE + 1 - in] = 0;
}

int WF_queuedTrace(__global const uint* queues, __global const uint* control, __global const int* wave, int i) {
    int in = wave[5];
    if (i >= (int) control[CONTROL_TRACE + in]) return -1;
    return (int) queues[(QUEUE_TRACE + in) * wave[3] + i];
}

// The hardware query for the next block of each queued trace.
__kernel void wf_trace(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT
    int p = WF_queuedTrace(queues, control, wave, GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    HwRtCandidate candidate;
    float distance;
    WF_I(s, TRACE_I + 4) += 1;
    bool found = HwRt_query(hwAccel, hwFunctions, WF_loadRay(s), WF_F(s, TRACE_F), HUGE_VALF, &candidate, &distance);
#ifdef CHUNKYCL_HWRT_STATS
    // [14] hardware queries, [15] intersection function calls they made
    ATOMIC_ADD(control + CONTROL_STATS + 2, 1u);
    ATOMIC_ADD(control + CONTROL_STATS + 3, (uint) candidate.calls);
#endif
    WF_I(s, TRACE_I + 9) = found ? 1 : 0;
    if (!found) {
#ifdef CHUNKYCL_HWRT_STATS
        ATOMIC_ADD(control + CONTROL_STATS, 1u);
        ATOMIC_ADD(control + CONTROL_STATS + 1, (uint) WF_I(s, TRACE_I + 4));
#endif
        return;
    }
    WF_I(s, TRACE_I + 5) = candidate.cell.x;
    WF_I(s, TRACE_I + 6) = candidate.cell.y;
    WF_I(s, TRACE_I + 7) = candidate.cell.z;
    WF_I(s, TRACE_I + 8) = candidate.data;
    WF_F(s, TRACE_F + 1) = distance;
    WF_F(s, TRACE_F + 2) = candidate.leafExit;
#endif
}

// Marches the octree from the block found by wf_trace, for wave[7] leaves (as many as needed in the
// last round): the next blocks are usually close, so this resolves most rays that miss the first
// block (a model that doesn't fill its block, a transparent pixel, the block the ray is inside of)
// without another hardware query. A hit is stored as the world octree hit; a path that found
// nothing in those leaves goes to the next round's queue.
__kernel void wf_resolve(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT
    int p = WF_queuedTrace(queues, control, wave, GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, TRACE_I + 9)) return;  // Nothing more along the ray

    WF_SCENE(scene);
    Ray ray = WF_loadRay(s);
    float start = WF_F(s, TRACE_F);
    int queries = WF_I(s, TRACE_I + 4);
    float distance = WF_F(s, TRACE_F + 1);

    // As closestIntersect starts the world octree test.
    IntersectionRecord record = IntersectionRecord_new();
    record.medium = ray.material;
    MaterialSample sample;

    // Blocks in between that can't be hit (air) end the block the ray was in.
    float3 ad = fabs(ray.origin);
    float eps = fmax(OFFSET, fmax(fmax(ad.x, ad.y), ad.z) * 4.8e-7f + distance * 4.8e-7f);
    int current = distance > start + eps ? 0 : WF_I(s, TRACE_I);
    bool last = wave[6] || queries >= 256;
    int march = Octree_march(scene.octree, textureAtlas, scene.blockPalette, scene.materialPalette,
                             last ? 256 : wave[7], HUGE_VALF, ray, &record, &sample, &distance, &current);
    if (march == MARCH_HIT) {
        WF_storeHit(s, true, record, sample);
    } else if (march == MARCH_MORE && !last) {
        WF_F(s, TRACE_F) = distance;
        WF_I(s, TRACE_I) = current;
        int out = 1 - wave[5];
        queues[(QUEUE_TRACE + out) * wave[3] + ATOMIC_ADD(control + CONTROL_TRACE + out, 1u)] = (uint) p;
    }
#ifdef CHUNKYCL_HWRT_STATS
    if (march != MARCH_MORE || last) {
        ATOMIC_ADD(scene.stats, 1u);
        ATOMIC_ADD(scene.stats + 1, (uint) queries);
    }
#endif
#endif
}

// Alternative to the rounds (-Dchunkymetal.traceMode=fused): the whole world octree trace with
// hardware ray tracing in one small kernel (after wf_begin), then wf_intersectRest.
__kernel void wf_world(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    WF_SCENE(scene);
    Ray ray = WF_loadRay(s);
    IntersectionRecord record = IntersectionRecord_new();
    record.medium = ray.material;
    MaterialSample sample;
    HwRtTrace trace = HwRt_begin(ray);
    if (HwRt_continue(hwAccel, hwFunctions, &trace, scene.octree, textureAtlas, scene.blockPalette,
                      scene.materialPalette, ray, &record, &sample)) {
        WF_storeHit(s, true, record, sample);
    }
#endif
}

// The rest of the intersection after the world octree (entities, clouds, water), like closestIntersect.
__kernel void wf_intersectRest(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    WF_SCENE(scene);
    Ray ray = WF_loadRay(s);
    IntersectionRecord record = IntersectionRecord_new();
    record.medium = ray.material;
    MaterialSample sample;
    bool hit = WF_I(s, HIT_I) != 0;
    if (hit) WF_loadHit(s, &record, &sample);
#ifdef CHUNKYCL_HWRT_WATER
    bool enterWater = false;  // wf_traceWater and wf_resolveWater
#else
    bool enterWater = true;
#endif
    hit = Scene_intersectAfterWorld(scene, textureAtlas, ray, ray, hit, enterWater, &record, &sample);
    WF_storeHit(s, hit, record, sample);
}

// The hardware query for the first water block along the ray of each queued path outside water,
// closer than what wf_intersectRest found (CHUNKYCL_HWRT_WATER).
__kernel void wf_traceWater(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT_WATER
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    WF_I(s, TRACE_I + 9) = 0;
    if (!WF_I(s, 3) || (WF_I(s, 1) & RAY_WATER)) return;
    float maxDistance = WF_I(s, HIT_I) ? WF_F(s, HIT_F) : HUGE_VALF;
    HwRtCandidate candidate;
    float distance;
    if (HwRt_query(hwWaterAccel, hwWaterFunctions, WF_loadRay(s), 0.0f, maxDistance, &candidate, &distance)) {
        WF_I(s, TRACE_I + 9) = 1;
        WF_F(s, TRACE_F + 1) = distance;
    }
#endif
}

// The water octree march (as in closestIntersect) from the water block found by wf_traceWater.
__kernel void wf_resolveWater(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
#ifdef CHUNKYCL_HWRT_WATER
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3) || !WF_I(s, TRACE_I + 9)) return;
    WF_SCENE(scene);
    WF_NO_HWRT(scene);
    Ray ray = WF_loadRay(s);
    IntersectionRecord record = IntersectionRecord_new();
    record.medium = ray.material;
    MaterialSample sample;
    bool hit = WF_I(s, HIT_I) != 0;
    if (hit) WF_loadHit(s, &record, &sample);

    float distance = WF_F(s, TRACE_F + 1);
    float3 ad = fabs(ray.origin);
    float eps = fmax(OFFSET, fmax(fmax(ad.x, ad.y), ad.z) * 4.8e-7f + distance * 4.8e-7f);
    // The march from the ray origin would have passed air before reaching this block.
    int current = distance > eps ? 0 : ray.material;
    if (Octree_march(scene.waterOctree, textureAtlas, scene.blockPalette, scene.materialPalette, 256, HUGE_VALF,
                     ray, &record, &sample, &distance, &current) == MARCH_HIT) {
        WF_storeHit(s, true, record, sample);
    }
#endif
}

// Russian roulette, then the closest intersection of the path's ray.
__kernel void wf_intersect(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    int depth = wave[2];

    // See the render kernel (rayTracer.c) for the survival chance.
    if (depth >= 3) {
        unsigned int randomState = (unsigned int) WF_I(s, 2);
        Random random = &randomState;
        float3 throughput = WF_get3(s, 9);
        float survival = fmin(1.0f, fmax(0.25f, fmax(throughput.x, fmax(throughput.y, throughput.z))));
        bool survives = Random_nextFloat(random) < survival;
        WF_I(s, 2) = (int) randomState;
        if (!survives) {
            WF_I(s, 3) = 0;
            return;
        }
        WF_set3(s, 9, throughput / survival);
    }

    WF_SCENE(scene);
    Ray ray = WF_loadRay(s);
    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;
    bool hit = closestIntersect(scene, textureAtlas, ray, &record, &sample, &material);
    WF_storeHit(s, hit, record, sample);
}

// Everything the render kernel does with an intersection, except the shadow ray of sun sampling.
__kernel void wf_shade(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = WF_queued(queues, control, wave, wave[4], GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, 3)) return;
    int depth = wave[2];
    WF_SCENE(scene);
    Sun sun = Sun_new(sunData);

    Ray ray = WF_loadRay(s);
    float3 color = WF_get3(s, 6);
    float3 throughput = WF_get3(s, 9);
    unsigned int randomState = (unsigned int) WF_I(s, 2);
    Random random = &randomState;
    bool alive = true;

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    if (WF_I(s, HIT_I)) {
        WF_loadHit(s, &record, &sample);
        Material material = (record.flags & HIT_CLOUD) ? Material_cloud() : Material_get(scene.materialPalette, record.material);
#ifdef CHUNKYCL_FOG
        if (ray.material == 0) {
            Scene_groundFog(scene, textureAtlas, sun, ray, record.distance, &color, &throughput, random);
        }
#endif
        if (ray.flags & RAY_WATER) {
            throughput *= Water_attenuation(scene.water, record.distance);
        }

        if (record.flags & HIT_WATER) {
            float3 hitPoint = ray.origin + ray.direction * record.distance;
            if (scene.water.waveMode != WAVES_STILL && record.normal.y != 0) {
                record.normal = Water_waveNormal(scene.water, hitPoint);
            }
            Ray incoming = ray;
            bool diffuse = Water_scatter(scene.water, record, sample, &ray, &throughput, random);
            bool transmitted = ((ray.flags ^ incoming.flags) & RAY_WATER) != 0;
            if (diffuse && (scene.sunStrategy & SUN_SAMPLING) && depth + 1 < *rayDepth) {
                WF_requestSunLight(scene, sun, incoming, record, random, throughput, s);
            }
            if (diffuse) ray.flags |= RAY_DIFFUSE;
            else if (!transmitted) ray.flags &= ~RAY_DIFFUSE;
            ray.origin = hitPoint + ray.direction * OFFSET;
        } else {
            float n1 = (ray.flags & RAY_WATER) ? scene.water.ior : DEFAULT_IOR;
            MaterialPdfSample pdfSample = Material_samplePdf(material, record, sample, ray, n1,
                    scene.water.fancierTranslucency, scene.water.transmissivityCap, random);
            throughput *= pdfSample.spectrum;

            bool diffuse = !pdfSample.specular && !pdfSample.transmitted;
            if (diffuse) {
                color += pdfSample.spectrum * sample.emittance * scene.emitterIntensity * throughput;
            }
            if (diffuse && (scene.sunStrategy & SUN_SAMPLING) && depth + 1 < *rayDepth) {
                WF_requestSunLight(scene, sun, ray, record, random, throughput, s);
            }
            if (diffuse) ray.flags |= RAY_DIFFUSE;
            else if (!pdfSample.transmitted) ray.flags &= ~RAY_DIFFUSE;

            if (pdfSample.transmitted) {
                float3 hitPoint = ray.origin + ray.direction * record.distance;
                ray.direction = pdfSample.direction;
                ray.origin = hitPoint + ray.direction * OFFSET;
                ray.material = record.block;
            } else {
                ray.origin = ray.origin + ray.direction * (record.distance - OFFSET);
                ray.direction = pdfSample.direction;
                ray.origin += ray.direction * OFFSET;
                ray.material = record.medium;
            }

            if (!pdfSample.specular) {
                ray.flags |= RAY_INDIRECT;
            }
        }
    } else {
        float skyMultiplier = depth == 0 ? skyLight[1] : skyLight[0];
        intersectSky(skyTexture, skyMultiplier, sun, textureAtlas, ray, Scene_sunMode(scene, ray), &sample);
#ifdef CHUNKYCL_FOG
        if (!(ray.flags & RAY_DIFFUSE)) {
            sample.color.xyz = Scene_skyFog(scene, textureAtlas, sun, ray, sample.color.xyz, random);
        }
#endif
        throughput *= sample.color.xyz;
        color += sample.emittance * throughput;
        alive = false;
    }

    WF_storeRay(s, ray);
    WF_set3(s, 6, color);
    WF_set3(s, 9, throughput);
    WF_I(s, 2) = (int) randomState;
    WF_I(s, 3) = alive ? 1 : 0;

    // Queue the path for the next bounce and for its sun light.
    if (alive) {
        int out = 1 - wave[4];
        queues[out * wave[3] + ATOMIC_ADD(control + out, 1u)] = (uint) p;
    }
    if (WF_I(s, SHADOW_I)) {
        queues[QUEUE_SHADOW * wave[3] + ATOMIC_ADD(control + QUEUE_SHADOW, 1u)] = (uint) p;
    }
}

// The sun light requested by wf_shade, through translucent blocks and water to the sun.
__kernel void wf_shadow(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = WF_queued(queues, control, wave, QUEUE_SHADOW, GLOBAL_ID);
    if (p < 0) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, SHADOW_I)) return;
    WF_I(s, SHADOW_I) = 0;
    WF_SCENE(scene);
#ifndef CHUNKYCL_HWRT_SHADOWS
    WF_NO_HWRT(scene);  // Shadow rays: hardware calls in this kernel cost more than they save
#endif

    Ray shadow;
    shadow.origin = WF_get3(s, SHADOW_F);
    shadow.direction = WF_get3(s, SHADOW_F + 3);
    shadow.material = WF_I(s, SHADOW_I + 1);
    shadow.flags = WF_I(s, SHADOW_I + 2);
    float3 attenuation = Scene_directLightAttenuation(scene, textureAtlas, shadow);
    WF_set3(s, 6, WF_get3(s, 6) + WF_get3(s, SHADOW_F + 6) * attenuation);
}

// Adds the paths' light to the pass buffer (the running average over the samples so far).
__kernel void wf_finish(WF_PARAMS KERNEL_GID_PARAM) {
    WF_BUFFERS;
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
    int gid = wave[0] + p;
    PathState s = WF_path(stateF, stateI, wave, p);
    float3 color = WF_get3(s, 6);
    int spp = *bufferSpp;
    float3 bufferColor = vload3(gid, res);
    bufferColor = (bufferColor * spp + color) / (spp + 1);
    vstore3(bufferColor, gid, res);
}
