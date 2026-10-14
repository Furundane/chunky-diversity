// Wavefront version of the render kernel (rayTracer.c): the path tracing loop split into small
// kernels that run once per bounce over all paths of a wave, so that the kernels which trace rays
// (wf_intersect, wf_shadow) are not burdened with the state of the whole path tracer.
//
// Per render pass and wave of paths: wf_generate, then for each bounce wf_intersect, wf_shade and
// wf_shadow, then wf_finish. The paths do the same work as in the render kernel, in the same order
// and with the same random numbers.

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
// field (field k of path p at k * wave[3] + p) so that neighboring paths read neighboring memory:
//   path   (floats 0..11):  origin (3), direction (3), color (3), throughput (3)
//          (ints 0..3):     ray material, ray flags, random state, alive
//   hit    (floats 12..25): distance, normal (3), texCoord (2), sample color (4), emittance, specular,
//                           metalness, roughness
//          (ints 4..8):     hit, material, flags, block, medium
//   shadow (floats 26..34): origin (3), direction (3), contribution (3)  (sun light requested by wf_shade)
//          (ints 9..11):    requested, ray material, ray flags
#define STATE_F 35
#define STATE_I 12
#define HIT_F 12
#define HIT_I 4
#define SHADOW_F 26
#define SHADOW_I 9

// Metal hardware ray tracing of the world octree (hwrt.h), compiled in with CHUNKYCL_HWRT.
#ifdef CHUNKYCL_HWRT
#define WF_HWRT_PARAMS primitive_acceleration_structure hwAccel, intersection_function_table<> hwFunctions,
#define WF_HWRT_SCENE(scene) scene.hwAccel = hwAccel; scene.hwFunctions = hwFunctions
#else
#define WF_HWRT_PARAMS
#define WF_HWRT_SCENE(scene)
#endif

// The parameters of every wavefront kernel: the render kernel's, then the path state and the wave
// ([0] first pixel, [1] number of paths, [2] bounce, [3] paths the state buffers hold).
#define WF_PARAMS \
    __global const int* projectorType, \
    __global const float* cameraSettings, \
    __global const int* octreeDepth, \
    __global const int* octreeData, \
    WF_HWRT_PARAMS \
    __global const int* waterOctreeDepth, \
    __global const int* waterOctreeData, \
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
    __global float* stateF, \
    __global int* stateI, \
    __global const int* wave

// The Scene of the render kernel. A macro, since the palettes keep pointers into the struct.
#define WF_SCENE(scene) \
    Scene scene; \
    scene.materialPalette = MaterialPalette_new(matPalette); \
    scene.octree = Octree_create(octreeData, *octreeDepth); \
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth); \
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
    WF_HWRT_SCENE(scene)

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
}

// Russian roulette, then the closest intersection of the path's ray.
__kernel void wf_intersect(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
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
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
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
}

// The sun light requested by wf_shade, through translucent blocks and water to the sun.
__kernel void wf_shadow(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
    PathState s = WF_path(stateF, stateI, wave, p);
    if (!WF_I(s, SHADOW_I)) return;
    WF_I(s, SHADOW_I) = 0;
    WF_SCENE(scene);

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
