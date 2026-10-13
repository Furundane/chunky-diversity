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

Ray ray_to_camera(
        const __global int* projectorType,
        const __global float* cameraSettings,
        const __global int* canvasConfig,
        int gid,
        Random random
) {
    Ray ray;
    if (*projectorType != -1) {
        float3 cameraPos = vload3(0, cameraSettings);
        float3 m1s = vload3(1, cameraSettings);
        float3 m2s = vload3(2, cameraSettings);
        float3 m3s = vload3(3, cameraSettings);

        int width = canvasConfig[0];
        int fullWidth = canvasConfig[2];
        int fullHeight = canvasConfig[3];
        int cropX = canvasConfig[4];
        int cropY = canvasConfig[5];

        float halfWidth = fullWidth / (2.0 * fullHeight);
        float invHeight = 1.0 / fullHeight;
        float x = -halfWidth + ((gid % width) + Random_nextFloat(random) + cropX) * invHeight;
        float y = -0.5 + ((gid / width) + Random_nextFloat(random) + cropY) * invHeight;

        switch (*projectorType) {
            case 0:
                ray = Camera_pinHole(x, y, random, cameraSettings + 12);
                break;
        }

        ray.direction = normalize(make_float3(
                dot(m1s, ray.direction),
                        dot(m2s, ray.direction),
                        dot(m3s, ray.direction)
        ));
        ray.origin = make_float3(
                dot(m1s, ray.origin),
                        dot(m2s, ray.origin),
                        dot(m3s, ray.origin)
        );

        ray.origin += cameraPos;
    } else {
        ray = Camera_preGenerated(cameraSettings, gid);
    }
    return ray;
}

__kernel void render(
    __global const int* projectorType,
    __global const float* cameraSettings,

    __global const int* octreeDepth,
    __global const int* octreeData,
    __global const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,

    image2d_t skyTexture,
    __global const float* skyLight,  // [0] sky seen by reflected rays, [1] seen directly (apparent)
    __global const int* sunData,
    __global const float* waterSettings,

    __global const int* randomSeed,
    __global const int* bufferSpp,
    __global const int* canvasConfig,
    __global const int* rayDepth,
    __global float* res
    KERNEL_GID_PARAM
) {
    int gid = GLOBAL_ID;

    Scene scene;
    scene.materialPalette = MaterialPalette_new(matPalette);
    scene.octree = Octree_create(octreeData, *octreeDepth);
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth);
    scene.water = WaterSettings_new(waterSettings);
    scene.sunStrategy = (int) waterSettings[9];
    scene.emitterIntensity = waterSettings[10];
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs, &scene.materialPalette);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs, &scene.materialPalette);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, &scene.materialPalette);
    scene.drawDepth = 256;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = *randomSeed + gid;
    Random random = &randomState;
    Random_nextState(random);
    Ray ray = ray_to_camera(projectorType, cameraSettings, canvasConfig, gid, random);

    ray.material = 0;
    ray.flags = 0;
    if (Octree_isInsideWater(scene.waterOctree, scene.blockPalette, ray.origin)) {
        ray.flags |= RAY_WATER;
    }

    float3 color = (float3) (0.0);
    float3 throughput = (float3) (1.0);

    for (int depth = 0; depth < *rayDepth; depth++) {
        // Russian roulette: from the third bounce on, end paths that carry little light with a probability
        // based on their throughput and boost the survivors to compensate, so the expected result is unchanged.
        // The survival chance is at least 0.25: dim paths (e.g. underwater) otherwise survive so rarely, and
        // get boosted so much, that they show up as speckles. Measured as free on a typical scene.
        if (depth >= 3) {
            float survival = fmin(1.0f, fmax(0.25f, fmax(throughput.x, fmax(throughput.y, throughput.z))));
            if (Random_nextFloat(random) >= survival) {
                break;
            }
            throughput /= survival;
        }

        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;

        if (closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            if (ray.flags & RAY_WATER) {
                throughput *= Water_attenuation(scene.water, record.distance);
            }

            if (record.flags & HIT_WATER) {
                float3 hitPoint = ray.origin + ray.direction * record.distance;
                Ray incoming = ray;
                bool diffuse = Water_scatter(scene.water, record, sample, &ray, &throughput, random);
                bool transmitted = ((ray.flags ^ incoming.flags) & RAY_WATER) != 0;
                if (diffuse && (scene.sunStrategy & SUN_SAMPLING) && depth + 1 < *rayDepth) {
                    color += throughput * Scene_sunDirectLight(scene, textureAtlas, sun, incoming, record, random);
                }
                // Like Chunky, a transmitted ray keeps the diffuse/specular state of the previous bounce.
                if (diffuse) ray.flags |= RAY_DIFFUSE;
                else if (!transmitted) ray.flags &= ~RAY_DIFFUSE;
                ray.origin = hitPoint + ray.direction * OFFSET;
                continue;
            }

            float n1 = (ray.flags & RAY_WATER) ? scene.water.ior : DEFAULT_IOR;
            MaterialPdfSample pdfSample = Material_samplePdf(material, record, sample, ray, n1,
                    scene.water.fancierTranslucency, scene.water.transmissivityCap, random);
            throughput *= pdfSample.spectrum;

            bool diffuse = !pdfSample.specular && !pdfSample.transmitted;
            // Emitters, like PathTracer.doDiffuseReflection: only on diffuse reflection, with Chunky's quadratic
            // emittance mapping (color squared; pdfSample.spectrum is the color and is already in the throughput).
            if (diffuse) {
                color += pdfSample.spectrum * sample.emittance * scene.emitterIntensity * throughput;
            }
            // Next event estimation: light from the sun reaching this point directly. Like Chunky, there
            // is no scattering (and so no sun sampling) at the last bounce.
            if (diffuse && (scene.sunStrategy & SUN_SAMPLING) && depth + 1 < *rayDepth) {
                color += throughput * Scene_sunDirectLight(scene, textureAtlas, sun, ray, record, random);
            }
            // Like Chunky, a transmitted ray keeps the diffuse/specular state of the previous bounce.
            if (diffuse) ray.flags |= RAY_DIFFUSE;
            else if (!pdfSample.transmitted) ray.flags &= ~RAY_DIFFUSE;

            if (pdfSample.transmitted) {
                // Continue on the far side of the surface, inside the block that was hit.
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
        } else {
            // Like Sky.getSkyColorInterpolated / getSkyColor in Chunky: camera rays see the apparent sky light.
            float skyMultiplier = depth == 0 ? skyLight[1] : skyLight[0];
            intersectSky(skyTexture, skyMultiplier, sun, textureAtlas, ray, Scene_sunMode(scene, ray), &sample);
            throughput *= sample.color.xyz;
            color += sample.emittance * throughput;
            break;
        }
    }

    int spp = *bufferSpp;
    float3 bufferColor = vload3(gid, res);
    bufferColor = (bufferColor * spp + color) / (spp + 1);
    vstore3(bufferColor, gid, res);
}

__kernel void preview(
    __global const int* projectorType,
    __global const float* cameraSettings,

    __global const int* octreeDepth,
    __global const int* octreeData,
    __global const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,

    image2d_t skyTexture,
    __global const float* skyLight,  // [0] sky seen by reflected rays, [1] seen directly (apparent)
    __global const int* sunData,
    __global const float* waterSettings,

    __global const int* canvasConfig,
    __global int* res
    KERNEL_GID_PARAM
) {
    int gid = GLOBAL_ID;

    int px = gid % canvasConfig[0] + canvasConfig[4];
    int py = gid / canvasConfig[0] + canvasConfig[5];

    // Crosshairs?
    if ((px == canvasConfig[2] / 2 && (py >= canvasConfig[3] / 2 - 5 && py <= canvasConfig[3] / 2 + 5)) ||
        (py == canvasConfig[3] / 2 && (px >= canvasConfig[2] / 2 - 5 && px <= canvasConfig[2] / 2 + 5))) {
        res[gid] = 0xFFFFFFFF;
        return;
    }

    Scene scene;
    scene.materialPalette = MaterialPalette_new(matPalette);
    scene.octree = Octree_create(octreeData, *octreeDepth);
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth);
    scene.water = WaterSettings_new(waterSettings);
    scene.sunStrategy = (int) waterSettings[9];
    scene.emitterIntensity = waterSettings[10];
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs, &scene.materialPalette);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs, &scene.materialPalette);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, &scene.materialPalette);
    scene.drawDepth = 256;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = 0;
    Random random = &randomState;
    Random_nextState(random);

    Ray ray = ray_to_camera(projectorType, cameraSettings, canvasConfig, gid, random);

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;

    ray.material = 0;
    ray.flags = RAY_PREVIEW;

    if (Octree_isInsideWater(scene.waterOctree, scene.blockPalette, ray.origin)) {
        ray.flags |= RAY_WATER;
    }

    float3 color;
    if (closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
        float shading = dot(record.normal, make_float3(0.25, 0.866, 0.433));
        shading = fmax(0.3f, shading);
        if (record.flags & HIT_WATER) {
            color = Water_color(scene.water, sample) * shading;
        } else {
            color = sample.color.xyz * shading;
        }
    } else {
        intersectSky(skyTexture, skyLight[1], sun, textureAtlas, ray, SUN_TEXTURE, &sample);
        color = sample.color.xyz;
    }

    color = sqrt(color);
    int3 rgb = intFloorFloat3(clamp(color * 255.0f, 0.0f, 255.0f));
    res[gid] = 0xFF000000 | (rgb.x << 16) | (rgb.y << 8) | rgb.z;
}
