// Water geometry and shading. Mirrors se.llbit.chunky.block.minecraft.Water and the water handling in
// se.llbit.chunky.renderer.scene.PathTracer (still water only, no wave shading).
#ifndef CHUNKYCLPLUGIN_WATER_H
#define CHUNKYCLPLUGIN_WATER_H

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "constants.h"
#include "material.h"
#include "random.h"

// Block palette model type for water. Palette entry: [MODEL_WATER, material pointer, water data]
#define MODEL_WATER 5
#define WATER_FULL_BLOCK (1 << 16)

typedef struct {
    float3 color;        // Linear water color: custom color, or the biome color multiplied with the texture
    float opacity;
    float visibility;
    float ior;
    bool useCustomColor;
    // Scene translucency settings, also used for other translucent materials
    float transmissivityCap;
    bool fancierTranslucency;
} WaterSettings;

WaterSettings WaterSettings_new(__global const float* data) {
    WaterSettings ws;
    ws.color = vload3(0, data);
    ws.opacity = data[3];
    ws.visibility = data[4];
    ws.ior = data[5];
    ws.useCustomColor = data[6] != 0;
    ws.transmissivityCap = data[7];
    ws.fancierTranslucency = data[8] != 0;
    return ws;
}

// Corner heights of a water block, see Water.height in Chunky.
typedef struct {
    float c0;  // SW (x=0, z=1)
    float c1;  // SE (x=1, z=1)
    float c2;  // NE (x=1, z=0)
    float c3;  // NW (x=0, z=0)
} WaterCorners;

float _Water_cornerHeight(int data, int shift) {
    int level = ((data >> shift) & 0xF) % 8;
    return (14.0f - 1.75f * level) / 16.0f;
}

WaterCorners WaterCorners_new(int data) {
    WaterCorners c;
    if (data & WATER_FULL_BLOCK) {
        c.c0 = c.c1 = c.c2 = c.c3 = 1;
    } else {
        c.c0 = _Water_cornerHeight(data, 0);
        c.c1 = _Water_cornerHeight(data, 4);
        c.c2 = _Water_cornerHeight(data, 8);
        c.c3 = _Water_cornerHeight(data, 12);
    }
    return c;
}

// Approximate surface height at a point inside the block.
float WaterCorners_heightAt(WaterCorners c, float x, float z) {
    return mix(mix(c.c3, c.c2, x), mix(c.c0, c.c1, x), z);
}

// Two-sided ray/triangle intersection. Returns the distance, or -1 on a miss.
float _Water_triangle(float3 o, float3 d, float3 a, float3 b, float3 c) {
    float3 e1 = b - a;
    float3 e2 = c - a;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (fabs(det) < 1e-9f) return -1;
    float inv = 1 / det;
    float3 s = o - a;
    float u = dot(s, p) * inv;
    if (u < 0 || u > 1) return -1;
    float3 q = cross(s, e1);
    float v = dot(d, q) * inv;
    if (v < 0 || u + v > 1) return -1;
    return dot(e2, q) * inv;
}

// Intersect the water surface (the two top triangles of Water.t012 / Water.t230) in block-local
// coordinates. Only hits with tmin < t < *t are accepted; the normal points up.
bool Water_intersectTop(WaterCorners c, float3 o, float3 d, float tmin, __private float* t, __private float3* normal) {
    bool hit = false;
    float3 a = make_float3(1, c.c1, 1), b = make_float3(1, c.c2, 0), e = make_float3(0, c.c0, 1);
    float tt = _Water_triangle(o, d, a, b, e);
    if (tt > tmin && tt < *t) {
        *t = tt;
        *normal = normalize(cross(b - a, e - a));
        hit = true;
    }
    a = make_float3(0, c.c3, 0); b = make_float3(0, c.c0, 1); e = make_float3(1, c.c2, 0);
    tt = _Water_triangle(o, d, a, b, e);
    if (tt > tmin && tt < *t) {
        *t = tt;
        *normal = normalize(cross(b - a, e - a));
        hit = true;
    }
    if (hit && normal->y < 0) *normal = -*normal;
    return hit;
}

// Side face test: the face lies on axis = value; the water edge height goes linearly from h0 to h1
// along the other horizontal axis.
void _Water_side(float3 o, float3 d, int axis, float value, float h0, float h1, float3 n,
                 __private float* t, __private float3* normal, __private float2* uv, __private bool* hit) {
    float dc = axis == 0 ? d.x : d.z;
    if (dc == 0) return;
    float oc = axis == 0 ? o.x : o.z;
    float tt = (value - oc) / dc;
    if (tt <= 0 || tt >= *t) return;
    float3 p = o + d * tt;
    float s = axis == 0 ? p.z : p.x;
    if (s < 0 || s > 1 || p.y < 0 || p.y > mix(h0, h1, s)) return;
    *t = tt;
    *normal = n;
    *uv = make_float2(s, p.y);
    *hit = true;
}

// Intersect the water volume of one block from the outside, in block-local coordinates.
bool Water_enter(WaterCorners c, float3 o, float3 d, float maxT, __private float* t, __private float3* normal, __private float2* uv) {
    bool hit = false;
    float best = maxT;
    float3 n;

    if (Water_intersectTop(c, o, d, 0, &best, &n)) {
        float3 p = o + d * best;
        *normal = n;
        *uv = make_float2(p.x, p.z);
        hit = true;
    }
    if (d.y != 0) {
        float tt = -o.y / d.y;
        float3 p = o + d * tt;
        if (tt > 0 && tt < best && p.x >= 0 && p.x <= 1 && p.z >= 0 && p.z <= 1) {
            best = tt;
            *normal = make_float3(0, -1, 0);
            *uv = make_float2(p.x, p.z);
            hit = true;
        }
    }
    _Water_side(o, d, 0, 0, c.c3, c.c0, make_float3(-1, 0, 0), &best, normal, uv, &hit);  // west
    _Water_side(o, d, 0, 1, c.c2, c.c1, make_float3(1, 0, 0), &best, normal, uv, &hit);   // east
    _Water_side(o, d, 2, 0, c.c3, c.c2, make_float3(0, 0, -1), &best, normal, uv, &hit);  // north
    _Water_side(o, d, 2, 1, c.c0, c.c1, make_float3(0, 0, 1), &best, normal, uv, &hit);   // south
    *t = best;
    return hit;
}

float3 Water_color(WaterSettings ws, MaterialSample sample) {
    return ws.useCustomColor ? ws.color : sample.color.xyz * ws.color;
}

// Water fog: attenuation over a distance travelled inside water.
float Water_attenuation(WaterSettings ws, float distance) {
    return ws.visibility == 0 ? 0 : exp(-distance / ws.visibility);
}

// Scatter a ray at a water surface (entering or leaving water). Follows PathTracer: specular
// reflection, then diffuse reflection off the water color (water opacity), then Fresnel
// reflection/refraction. Updates the ray direction, the RAY_WATER flag and the throughput.
bool Water_scatter(WaterSettings ws, IntersectionRecord record, MaterialSample sample, __private Ray* ray,
                   __private float3* throughput, Random random) {
    float3 d = ray->direction;
    float3 n = record.normal;
    if (dot(n, d) > 0) n = -n;  // Face the incoming ray
    IntersectionRecord facing = record;
    facing.normal = n;

    if (sample.specular > 0 && Random_nextFloat(random) < sample.specular) {
        ray->direction = _Material_specularReflection(facing, sample, *ray, random);
        return false;
    }

    float3 color = Water_color(ws, sample);
    float maxColor = fmax(color.x, fmax(color.y, color.z));
    float pDiffuse = ws.fancierTranslucency ? 1 - pow(1 - ws.opacity, maxColor) : ws.opacity;
    float pAbsorb = ws.fancierTranslucency ? 1 - (1 - ws.opacity) / (1 - pDiffuse + EPS) : ws.opacity;

    if (Random_nextFloat(random) < pDiffuse) {
        ray->direction = _Material_diffuseReflection(facing, random);
        *throughput *= color;
        ray->flags |= RAY_INDIRECT;
        return true;
    }

    float n1n2 = (ray->flags & RAY_WATER) ? ws.ior : 1 / ws.ior;
    float cosTheta = -dot(n, d);
    float radicand = 1 - n1n2 * n1n2 * (1 - cosTheta * cosTheta);
    float3 reflected = d - 2 * dot(d, n) * n;
    if (radicand < EPS) {
        // Total internal reflection
        ray->direction = reflected;
        return false;
    }

    float a = n1n2 - 1;
    float b = n1n2 + 1;
    float r0 = a * a / (b * b);
    float c = 1 - cosTheta;
    float fresnel = r0 + (1 - r0) * c * c * c * c * c;
    if (Random_nextFloat(random) < fresnel) {
        ray->direction = reflected;
        return false;
    }

    ray->direction = normalize(n1n2 * d + (n1n2 * cosTheta - sqrt(radicand)) * n);

    *throughput *= Material_transmission(color, pAbsorb, ws.fancierTranslucency, ws.transmissivityCap);
    ray->flags ^= RAY_WATER;
    return false;
}

#endif
