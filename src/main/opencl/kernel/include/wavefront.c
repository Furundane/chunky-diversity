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

// Path state, one entry per path of the wave:
//   pathF:   origin (3), direction (3), color (3), throughput (3)
//   pathI:   ray material, ray flags, random state, alive
//   hitF:    distance, normal (3), texCoord (2), sample color (4), emittance, specular, metalness, roughness
//   hitI:    hit, material, flags, block, medium
//   shadowF: origin (3), direction (3), contribution (3)  (sun light requested by wf_shade)
//   shadowI: requested, ray material, ray flags
#define PATH_F 12
#define PATH_I 4
#define HIT_F 14
#define HIT_I 5
#define SHADOW_F 9
#define SHADOW_I 3

// The parameters of every wavefront kernel: the render kernel's, then the path state and the wave
// ([0] first pixel, [1] number of paths, [2] bounce).
#define WF_PARAMS \
    __global const int* projectorType, \
    __global const float* cameraSettings, \
    __global const int* octreeDepth, \
    __global const int* octreeData, \
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
    __global float* pathF, \
    __global int* pathI, \
    __global float* hitF, \
    __global int* hitI, \
    __global float* shadowF, \
    __global int* shadowI, \
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
    scene.drawDepth = 256

Ray WF_loadRay(__global const float* pathF, __global const int* pathI, int p) {
    Ray ray;
    ray.origin = vload3(0, pathF + p * PATH_F);
    ray.direction = vload3(0, pathF + p * PATH_F + 3);
    ray.material = pathI[p * PATH_I];
    ray.flags = pathI[p * PATH_I + 1];
    return ray;
}

void WF_storeRay(__global float* pathF, __global int* pathI, int p, Ray ray) {
    vstore3(ray.origin, 0, pathF + p * PATH_F);
    vstore3(ray.direction, 0, pathF + p * PATH_F + 3);
    pathI[p * PATH_I] = ray.material;
    pathI[p * PATH_I + 1] = ray.flags;
}

void WF_storeHit(__global float* hitF, __global int* hitI, int p, bool hit, IntersectionRecord record, MaterialSample sample) {
    __global float* f = hitF + p * HIT_F;
    __global int* i = hitI + p * HIT_I;
    i[0] = hit ? 1 : 0;
    if (!hit) return;
    f[0] = record.distance;
    vstore3(record.normal, 0, f + 1);
    f[4] = record.texCoord.x;
    f[5] = record.texCoord.y;
    f[6] = sample.color.x;
    f[7] = sample.color.y;
    f[8] = sample.color.z;
    f[9] = sample.color.w;
    f[10] = sample.emittance;
    f[11] = sample.specular;
    f[12] = sample.metalness;
    f[13] = sample.roughness;
    i[1] = record.material;
    i[2] = record.flags;
    i[3] = record.block;
    i[4] = record.medium;
}

void WF_loadHit(__global const float* hitF, __global const int* hitI, int p, __private IntersectionRecord* record, __private MaterialSample* sample) {
    __global const float* f = hitF + p * HIT_F;
    __global const int* i = hitI + p * HIT_I;
    record->distance = f[0];
    record->normal = vload3(0, f + 1);
    record->texCoord = make_float2(f[4], f[5]);
    sample->color = make_float4(f[6], f[7], f[8], f[9]);
    sample->emittance = f[10];
    sample->specular = f[11];
    sample->metalness = f[12];
    sample->roughness = f[13];
    record->material = i[1];
    record->flags = i[2];
    record->block = i[3];
    record->medium = i[4];
}

// Scene_sunDirectLight without the shadow ray: the ray and the light it would add are left for
// wf_shadow, which multiplies the light by the attenuation along the ray.
void WF_requestSunLight(Scene self, Sun sun, Ray ray, IntersectionRecord record, Random random, float3 throughput,
                        __global float* shadowF, __global int* shadowI, int p) {
    float3 normal = record.normal;
    if (dot(normal, ray.direction) > 0) normal = -normal;
    float3 sunDirection = Sun_sampleDirection(sun, random);
    float cosine = dot(sunDirection, normal);
    if (cosine <= 0) {
        return;
    }
    float3 origin = ray.origin + ray.direction * record.distance + normal * OFFSET;
    float mult = cosine * ((self.sunStrategy & SUN_LUMINOSITY) ? sun.luminosityPdf : 1);
    vstore3(origin, 0, shadowF + p * SHADOW_F);
    vstore3(sunDirection, 0, shadowF + p * SHADOW_F + 3);
    vstore3(throughput * mult * sun.emittance, 0, shadowF + p * SHADOW_F + 6);
    shadowI[p * SHADOW_I] = 1;
    shadowI[p * SHADOW_I + 1] = record.medium;
    shadowI[p * SHADOW_I + 2] = ray.flags & RAY_WATER;
}

__kernel void wf_generate(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
    int gid = wave[0] + p;
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
    WF_storeRay(pathF, pathI, p, ray);
    vstore3(make_float3(0, 0, 0), 0, pathF + p * PATH_F + 6);
    vstore3(make_float3(1, 1, 1), 0, pathF + p * PATH_F + 9);
    pathI[p * PATH_I + 2] = (int) randomState;
    pathI[p * PATH_I + 3] = 1;
    shadowI[p * SHADOW_I] = 0;
}

// Russian roulette, then the closest intersection of the path's ray.
__kernel void wf_intersect(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1] || !pathI[p * PATH_I + 3]) return;
    int depth = wave[2];

    // See the render kernel (rayTracer.c) for the survival chance.
    if (depth >= 3) {
        unsigned int randomState = (unsigned int) pathI[p * PATH_I + 2];
        Random random = &randomState;
        float3 throughput = vload3(0, pathF + p * PATH_F + 9);
        float survival = fmin(1.0f, fmax(0.25f, fmax(throughput.x, fmax(throughput.y, throughput.z))));
        bool survives = Random_nextFloat(random) < survival;
        pathI[p * PATH_I + 2] = (int) randomState;
        if (!survives) {
            pathI[p * PATH_I + 3] = 0;
            return;
        }
        vstore3(throughput / survival, 0, pathF + p * PATH_F + 9);
    }

    WF_SCENE(scene);
    Ray ray = WF_loadRay(pathF, pathI, p);
    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;
    bool hit = closestIntersect(scene, textureAtlas, ray, &record, &sample, &material);
    WF_storeHit(hitF, hitI, p, hit, record, sample);
}

// Everything the render kernel does with an intersection, except the shadow ray of sun sampling.
__kernel void wf_shade(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1] || !pathI[p * PATH_I + 3]) return;
    int depth = wave[2];
    WF_SCENE(scene);
    Sun sun = Sun_new(sunData);

    Ray ray = WF_loadRay(pathF, pathI, p);
    float3 color = vload3(0, pathF + p * PATH_F + 6);
    float3 throughput = vload3(0, pathF + p * PATH_F + 9);
    unsigned int randomState = (unsigned int) pathI[p * PATH_I + 2];
    Random random = &randomState;
    bool alive = true;

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    if (hitI[p * HIT_I]) {
        WF_loadHit(hitF, hitI, p, &record, &sample);
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
                WF_requestSunLight(scene, sun, incoming, record, random, throughput, shadowF, shadowI, p);
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
                WF_requestSunLight(scene, sun, ray, record, random, throughput, shadowF, shadowI, p);
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

    WF_storeRay(pathF, pathI, p, ray);
    vstore3(color, 0, pathF + p * PATH_F + 6);
    vstore3(throughput, 0, pathF + p * PATH_F + 9);
    pathI[p * PATH_I + 2] = (int) randomState;
    pathI[p * PATH_I + 3] = alive ? 1 : 0;
}

// The sun light requested by wf_shade, through translucent blocks and water to the sun.
__kernel void wf_shadow(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1] || !shadowI[p * SHADOW_I]) return;
    shadowI[p * SHADOW_I] = 0;
    WF_SCENE(scene);

    Ray shadow;
    shadow.origin = vload3(0, shadowF + p * SHADOW_F);
    shadow.direction = vload3(0, shadowF + p * SHADOW_F + 3);
    shadow.material = shadowI[p * SHADOW_I + 1];
    shadow.flags = shadowI[p * SHADOW_I + 2];
    float3 attenuation = Scene_directLightAttenuation(scene, textureAtlas, shadow);
    float3 color = vload3(0, pathF + p * PATH_F + 6);
    color += vload3(0, shadowF + p * SHADOW_F + 6) * attenuation;
    vstore3(color, 0, pathF + p * PATH_F + 6);
}

// Adds the paths' light to the pass buffer (the running average over the samples so far).
__kernel void wf_finish(WF_PARAMS KERNEL_GID_PARAM) {
    int p = GLOBAL_ID;
    if (p >= wave[1]) return;
    int gid = wave[0] + p;
    float3 color = vload3(0, pathF + p * PATH_F + 6);
    int spp = *bufferSpp;
    float3 bufferColor = vload3(gid, res);
    bufferColor = (bufferColor * spp + color) / (spp + 1);
    vstore3(bufferColor, gid, res);
}
