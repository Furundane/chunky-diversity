#ifndef CHUNKYCL_KERNEL_H
#define CHUNKYCL_KERNEL_H

#include "../opencl.h"
#include "rt.h"
#include "octree.h"
#include "block.h"
#include "constants.h"
#include "bvh.h"
#include "sky.h"
#include "water.h"


typedef struct {
    Octree octree;
    Octree waterOctree;
    Bvh worldBvh;
    Bvh actorBvh;
    BlockPalette blockPalette;
    MaterialPalette materialPalette;
    WaterSettings water;
    int sunStrategy;  // SUN_* flags from the scene's sun sampling strategy
    int drawDepth;
} Scene;

// Sun sampling strategy flags, see SunSamplingStrategy in Chunky
#define SUN_SAMPLING      0b0001  // Sample the sun as direct light on diffuse reflections
#define SUN_DIFFUSE_SKY   0b0010  // Diffuse reflections that escape to the sky see the sun
#define SUN_STRICT_DIRECT 0b0100  // Direct light is blocked by surfaces that change the index of refraction
#define SUN_LUMINOSITY    0b1000  // Scale direct light by the luminosity pdf


bool closestIntersect(Scene self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    float distance = 0;

    for (int i = 0; i < self.drawDepth; i++) {
        IntersectionRecord tempRecord = *record;
        tempRecord.distance = record->distance - distance;
        tempRecord.medium = ray.material;

        Ray tempRay = ray;
        tempRay.origin += distance * tempRay.direction;

        if (tempRecord.distance <= 0) {
            return false;
        }

        bool hit = false;
        hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
        hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, tempRay, &tempRecord, sample);
        hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, tempRay, &tempRecord, sample);

        // Water is tested last so that only a closer water surface sets HIT_WATER.
        if (ray.flags & RAY_WATER) {
            hit |= Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
        } else {
            hit |= Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
        }

        if (!hit) {
            return false;
        }

        distance += tempRecord.distance;

        *record = tempRecord;
        record->distance = distance;
        *mat = Material_get(self.materialPalette, record->material);
        return true;
    }
    return false;
}

void intersectSky(image2d_t skyTexture, float skyIntensity, Sun sun, image2d_array_t atlas, Ray ray, int sunMode, MaterialSample* sample) {
    Sky_intersect(skyTexture, skyIntensity, ray, sample);
    Sun_intersect(sun, atlas, ray, sunMode, sample);
}

// How a ray escaping to the sky sees the sun, see PathTracer.pathTrace.
int Scene_sunMode(Scene self, Ray ray) {
    if (!(ray.flags & RAY_DIFFUSE)) return SUN_TEXTURE;
    return (self.sunStrategy & SUN_DIFFUSE_SKY) ? SUN_DIFFUSE : SUN_NONE;
}

// Fraction of sunlight that reaches the ray origin along the ray, through translucent surfaces and water.
// See PathTracer.getDirectLightAttenuation.
float3 Scene_directLightAttenuation(Scene self, image2d_array_t atlas, Ray ray) {
    float3 attenuation = (float3) (1);
    float w = 1;
    for (int i = 0; i < 16; i++) {
        ray.origin += ray.direction * OFFSET;
        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;
        if (!closestIntersect(self, atlas, ray, &record, &sample, &material)) {
            break;
        }
        if (ray.flags & RAY_WATER) {
            w *= Water_attenuation(self.water, record.distance);
        }
        float3 color;
        float alpha;
        if (record.flags & HIT_WATER) {
            color = Water_color(self.water, sample);
            alpha = self.water.opacity;
            ray.flags ^= RAY_WATER;
            if (self.sunStrategy & SUN_STRICT_DIRECT) w = 0;
        } else {
            color = sample.color.xyz;
            alpha = sample.color.w;
            if ((self.sunStrategy & SUN_STRICT_DIRECT) && material.ior != DEFAULT_IOR) w = 0;
            ray.material = record.block;
        }
        float mult = 1 - alpha;
        attenuation *= color * alpha + mult;
        w *= mult;
        if (w <= 0) {
            return (float3) (0);
        }
        ray.origin += ray.direction * record.distance;
    }
    return attenuation * w;
}

// Direct sunlight arriving at a diffuse surface hit by the ray (before the surface color is applied).
// See the sun sampling in PathTracer.doDiffuseReflection.
float3 Scene_sunDirectLight(Scene self, image2d_array_t atlas, Sun sun, Ray ray, IntersectionRecord record, Random random) {
    float3 normal = record.normal;
    if (dot(normal, ray.direction) > 0) normal = -normal;
    float3 sunDirection = Sun_sampleDirection(sun, random);
    float cosine = dot(sunDirection, normal);
    if (cosine <= 0) {
        return (float3) (0);
    }
    Ray shadow;
    shadow.origin = ray.origin + ray.direction * record.distance + normal * OFFSET;
    shadow.direction = sunDirection;
    shadow.material = record.medium;
    shadow.flags = ray.flags & RAY_WATER;
    float3 attenuation = Scene_directLightAttenuation(self, atlas, shadow);
    float mult = cosine * ((self.sunStrategy & SUN_LUMINOSITY) ? sun.luminosityPdf : 1);
    return attenuation * mult * sun.emittance;
}

#endif
