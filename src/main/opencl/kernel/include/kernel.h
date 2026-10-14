#ifndef CHUNKYCL_KERNEL_H
#define CHUNKYCL_KERNEL_H

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "octree.h"
#include "block.h"
#include "constants.h"
#include "bvh.h"
#include "sky.h"
#include "water.h"
#include "atmosphere.h"
#ifdef CHUNKYCL_HWRT
#include "hwrt.h"
#endif


typedef struct {
    Octree octree;
    Octree waterOctree;
    Bvh worldBvh;
    Bvh actorBvh;
    BlockPalette blockPalette;
    MaterialPalette materialPalette;
    WaterSettings water;
    // Fog and clouds (atmosphere.h). Only these are kept here; the rest is read from the buffer when
    // needed, since the scene is passed around by value and a larger struct slows the kernel down.
    __global const int* atmosphere;
    int fogMode;
    bool clouds;
    int sunStrategy;  // SUN_* flags from the scene's sun sampling strategy
    float emitterIntensity;  // Scene emitter intensity, 0 when emitters are disabled
    int drawDepth;
#ifdef CHUNKYCL_HWRT  // Metal hardware ray tracing of the world octree (hwrt.h)
    primitive_acceleration_structure hwAccel;
    intersection_function_table<> hwFunctions;
    // false: use the octree march anyway. Set to a constant in a kernel, so that the compiler drops
    // the unused path from it (hardware calls slow down the whole kernel they are in).
    bool hwWorld;
#endif
#ifdef CHUNKYCL_HWRT_STATS
    __global uint* stats;
#endif
} Scene;

// Sun sampling strategy flags, see SunSamplingStrategy in Chunky
#define SUN_SAMPLING      0b0001  // Sample the sun as direct light on diffuse reflections
#define SUN_DIFFUSE_SKY   0b0010  // Diffuse reflections that escape to the sky see the sun
#define SUN_STRICT_DIRECT 0b0100  // Direct light is blocked by surfaces that change the index of refraction
#define SUN_LUMINOSITY    0b1000  // Scale direct light by the luminosity pdf


// CloudMaterial: opaque white.
Material Material_cloud() {
    Material cloud;
    cloud.flags = 0;
    cloud.tint = 0;
    cloud.textureSize = 0;
    cloud.color = 0xFFFFFFFF;
    cloud.normal_emittance = 0;
    cloud.specular_metalness_roughness = 0;
    cloud.ior = DEFAULT_IOR;
    return cloud;
}

// The part of closestIntersect after the world octree: entities, clouds and water, within the
// distance of the world octree hit (hit: whether there is one, in record). tempRay is the ray from
// where this intersection starts; ray decides how water is treated.
bool Scene_intersectAfterWorld(Scene self, image2d_array_t atlas, Ray tempRay, Ray ray, bool hit,
                               __private IntersectionRecord* record, __private MaterialSample* sample) {
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, tempRay, record, sample);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, tempRay, record, sample);

#ifdef CHUNKYCL_CLOUDS  // Only compiled in when the scene has clouds (see AtmosphereData.kernelDefines)
    // Clouds come after the blocks, which do not set the record flags, and before water, which does.
    float cloudT;
    float3 cloudNormal;
    if (self.clouds && Clouds_intersect(self.atmosphere, tempRay.origin, tempRay.direction, record->distance, &cloudT, &cloudNormal)) {
        record->distance = cloudT;
        record->normal = cloudNormal;
        record->texCoord = make_float2(0, 0);
        record->material = 0;
        record->block = 0;
        record->flags = HIT_CLOUD;
        // CloudMaterial: opaque white
        sample->color = make_float4(1, 1, 1, 1);
        sample->emittance = 0;
        sample->specular = 0;
        sample->metalness = 0;
        sample->roughness = 0;
        hit = true;
    }
#endif

    // Water is tested last so that only a closer water surface sets HIT_WATER.
    if (ray.flags & RAY_WATER) {
        hit |= Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, record, sample);
    } else {
        hit |= Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, record, sample);
    }
    return hit;
}

bool closestIntersect(Scene self, image2d_array_t atlas, Ray ray, __private IntersectionRecord* record, __private MaterialSample* sample, __private Material* mat) {
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
#if defined(CHUNKYCL_HWRT) && defined(CHUNKYCL_HWRT_PROBE)
        {   // Diagnostic: one hardware call with its result unused, then the octree march
            intersector<> isect;
            isect.assume_geometry_type(geometry_type::bounding_box);
            HwRtCandidate candidate;
            metal::raytracing::ray r(tempRay.origin, tempRay.direction, 0.0f, tempRecord.distance);
            intersection_result<> found = isect.intersect(r, self.hwAccel, self.hwFunctions, candidate);
            tempRecord.flags |= (int) (found.distance < -1.0f);
        }
        hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
#elif defined(CHUNKYCL_HWRT)
        if (self.hwWorld) {
            int hwCalls = 0;
            hit |= HwRt_intersect(self.hwAccel, self.hwFunctions, self.octree, atlas, self.blockPalette, self.materialPalette, tempRay, &tempRecord, sample, &hwCalls);
#ifdef CHUNKYCL_HWRT_STATS  // Diagnostic: world intersections and hardware calls (wavefront.c)
            ATOMIC_ADD(self.stats, 1u);
            ATOMIC_ADD(self.stats + 1, (uint) hwCalls);
#endif
        } else {
            hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
        }
#else
        hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &tempRecord, sample);
#endif
        hit = Scene_intersectAfterWorld(self, atlas, tempRay, ray, hit, &tempRecord, sample);

        if (!hit) {
            return false;
        }

        distance += tempRecord.distance;

        *record = tempRecord;
        record->distance = distance;
        if (record->flags & HIT_CLOUD) {
            *mat = Material_cloud();
        } else {
            *mat = Material_get(self.materialPalette, record->material);
        }
        return true;
    }
    return false;
}

void intersectSky(__global const float* skyTexture, float skyMultiplier, Sun sun, image2d_array_t atlas, Ray ray, int sunMode, __private MaterialSample* sample) {
    Sky_intersect(skyTexture, skyMultiplier, ray, sample);
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

// Sunlight scattered by fog at a point: how much of the sun is visible from there (PathTracer samples a
// random sun direction and calls getDirectLightAttenuation). The sun emittance is not applied, like Chunky.
float3 Scene_fogScatter(Scene self, image2d_array_t atlas, Sun sun, float3 point, Random random) {
    Ray atmos;
    atmos.origin = point;
    atmos.direction = Sun_sampleDirection(sun, random);
    atmos.material = 0;
    atmos.flags = 0;
    return Scene_directLightAttenuation(self, atlas, atmos);
}

// Fog over a ray segment through air or water that ends at a surface (Fog.addGroundFog): light from
// beyond the segment is attenuated and fog-colored sunlight is scattered in.
void Scene_groundFog(Scene self, image2d_array_t atlas, Sun sun, Ray ray, float distance,
                     __private float3* color, __private float3* throughput, Random random) {
    if (self.fogMode == FOG_NONE || !(distance > 0)) {
        return;
    }
    Atmosphere a = Atmosphere_new(self.atmosphere);
    float extinction;
    float offset;
    float inscatter;
    if (a.fogMode == FOG_UNIFORM) {
        float density = a.uniformDensity * FOG_EXTINCTION_FACTOR;
        extinction = exp(-distance * density);
        offset = clamp(distance * Random_nextFloat(random), EPS, fmax(EPS, distance - EPS));
        inscatter = a.fastFog ? 1 - extinction : distance * density * exp(-offset * density);
    } else {
        float y1 = ray.origin.y;
        float y2 = y1 + ray.direction.y * distance;
        extinction = exp(Fog_layeredIntegral(a, y1, y2) / Fog_clampDy(ray.direction.y));
        offset = Fog_sampleLayeredOffset(a, random, y1, y2, ray.direction.y);
        inscatter = 1 - extinction;
    }
    float3 scatter = Scene_fogScatter(self, atlas, sun, ray.origin + ray.direction * offset, random);
    *color += *throughput * scatter * a.fogColor * inscatter;
    *throughput *= extinction;
}

// Fog in front of the sky, for rays that see the sky directly or through specular reflections
// (Fog.addSkyFog).
float3 Scene_skyFog(Scene self, image2d_array_t atlas, Sun sun, Ray ray, float3 skyColor, Random random) {
    if (self.fogMode == FOG_NONE) {
        return skyColor;
    }
    Atmosphere a = Atmosphere_new(self.atmosphere);
    if (a.fogMode == FOG_UNIFORM) {
        if (a.uniformDensity <= 0) {
            return skyColor;
        }
        float fog = 1;
        if (ray.direction.y > 0) {
            fog = 1 - ray.direction.y;
            fog *= fog;
        }
        fog *= a.skyFogDensity;
        return (1 - fog) * skyColor + fog * a.fogColor;
    } else if (a.fogMode == FOG_LAYERED) {
        float dy = ray.direction.y;
        float y1 = ray.origin.y;
        float y2 = y1 + dy * FOG_LIMIT;
        float extinction = exp(Fog_layeredIntegral(a, y1, y2) / Fog_clampDy(dy));
        float offset = Fog_sampleLayeredOffset(a, random, y1, dy > 0 ? a.yMax : a.yMin, dy);
        float3 scatter = Scene_fogScatter(self, atlas, sun, ray.origin + ray.direction * offset, random);
        return skyColor * extinction + (1 - extinction) * scatter * a.fogColor;
    }
    return skyColor;
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
