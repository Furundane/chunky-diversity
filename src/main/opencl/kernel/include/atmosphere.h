// Fog and clouds. Mirrors se.llbit.chunky.renderer.scene.Fog and the cloud code in Sky
// (cloudIntersection). The data layout is described in AtmosphereData.java.

#ifndef CHUNKYCL_ATMOSPHERE_H
#define CHUNKYCL_ATMOSPHERE_H

#include "../opencl.h"
#include "compat.h"
#include "constants.h"
#include "random.h"

#define FOG_NONE    0
#define FOG_UNIFORM 1
#define FOG_LAYERED 2

#define ATMOSPHERE_FOG_LAYERS 32
#define ATMOSPHERE_CLOUD_BITS (ATMOSPHERE_FOG_LAYERS + 4 * 16)

#define FOG_EXTINCTION_FACTOR 0.04f  // Fog.EXTINCTION_FACTOR
#define FOG_LIMIT 30000.0f           // Fog.FOG_LIMIT
#define CLOUD_THICKNESS 5.0f

typedef struct {
    __global const int* data;
    int fogMode;
    float uniformDensity;
    float skyFogDensity;
    bool fastFog;
    float3 fogColor;
    float yMin;
    float yMax;
    int fogLayers;
    bool clouds;
    float cloudSize;
    float3 cloudOffset;  // x and z in cloud cells, y (cloud bottom) in blocks
    float3 origin;       // Scene origin: clouds are placed in world coordinates
} Atmosphere;

Atmosphere Atmosphere_new(__global const int* data) {
    Atmosphere a;
    a.data = data;
    a.fogMode = data[0];
    a.uniformDensity = as_float(data[1]);
    a.skyFogDensity = as_float(data[2]);
    a.fastFog = data[3] != 0;
    a.fogColor = make_float3(as_float(data[4]), as_float(data[5]), as_float(data[6]));
    a.yMin = as_float(data[7]);
    a.yMax = as_float(data[8]);
    a.fogLayers = data[9];
    a.clouds = data[10] != 0;
    a.cloudSize = as_float(data[11]);
    a.cloudOffset = make_float3(as_float(data[12]), as_float(data[13]), as_float(data[14]));
    a.origin = make_float3(as_float(data[15]), as_float(data[16]), as_float(data[17]));
    return a;
}

// ---- Fog

// Fog.clampDy: avoids dividing by (almost) zero for horizontal rays.
float Fog_clampDy(float dy) {
    const float epsilon = 0.00001f;
    if (dy > 0) {
        return dy < epsilon ? epsilon : dy;
    }
    return dy > -epsilon ? -epsilon : dy;
}

// Integral of the layered fog density between heights y1 and y2 (Fog.addLayeredFog): the layers have a
// logistic density profile, so this is a difference of logistic CDFs.
float Fog_layeredIntegral(Atmosphere a, float y1, float y2) {
    float total = 0;
    for (int i = 0; i < a.fogLayers; i++) {
        __global const int* layer = a.data + ATMOSPHERE_FOG_LAYERS + 4 * i;
        float y = as_float(layer[0]);
        float breadthInv = as_float(layer[2]);
        float density = as_float(layer[3]);
        total += density * (1 / (1 + exp((y - y1) * breadthInv)) - 1 / (1 + exp((y - y2) * breadthInv)));
    }
    return total;
}

// Distance along the ray to a random point of a layered fog segment from y1 to y2, distributed like the
// fog density of the first layer (Fog.sampleLayeredScatterOffset).
float Fog_sampleLayeredOffset(Atmosphere a, Random random, float y1, float y2, float dy) {
    if (a.fogLayers == 0) {
        return EPS;
    }
    __global const int* layer = a.data + ATMOSPHERE_FOG_LAYERS;
    float y = as_float(layer[0]);
    float breadth = as_float(layer[1]);
    float breadthInv = as_float(layer[2]);
    float y1v = 1 / (1 + exp((y - y1) * breadthInv));
    float y2v = 1 / (1 + exp((y - y2) * breadthInv));
    float middle = y1v + Random_nextFloat(random) * (y2v - y1v);
    float offsetY = log(middle / (1 - middle)) * breadth + y;
    return (offsetY - y1) / Fog_clampDy(dy);
}

// ---- Clouds

// Clouds.getCloud: the 256 x 256 cloud bitmap repeats in both directions.
bool Clouds_get(Atmosphere a, int x, int z) {
    x = ((x % 256) + 256) % 256;
    z = ((z % 256) + 256) % 256;
    int bit = z * 256 + x;
    return ((a.data[ATMOSPHERE_CLOUD_BITS + (bit >> 5)] >> (bit & 31)) & 1) != 0;
}

// First cloud surface along the ray closer than maxT (Sky.cloudIntersection). Clouds are boxes of
// cloudSize x 5 x cloudSize blocks. A ray that starts inside a cloud hits the surface where it leaves it.
bool Clouds_intersect(__global const int* data, float3 o, float3 d, float maxT, __private float* t, __private float3* normal) {
    Atmosphere a = Atmosphere_new(data);
    float ox = o.x + a.origin.x;
    float oy = o.y + a.origin.y;
    float oz = o.z + a.origin.z;
    float invSize = 1 / a.cloudSize;
    float bottom = a.cloudOffset.y;
    float top = bottom + CLOUD_THICKNESS;

    bool inside = false;
    float tStart = 0;
    if (oy < bottom || oy > top) {
        if (fabs(d.y) < 0.00001f) {
            return false;  // Almost horizontal, see Chunky issue #1664
        }
        tStart = d.y > 0 ? (bottom - oy) / d.y : (top - oy) / d.y;
        if (tStart < 0 || tStart >= maxT) {
            return false;
        }
        float cx = (ox + d.x * tStart) * invSize + a.cloudOffset.x;
        float cz = (oz + d.z * tStart) * invSize + a.cloudOffset.z;
        if (Clouds_get(a, (int) floor(cx), (int) floor(cz))) {
            *t = tStart;
            *normal = make_float3(0, d.y > 0 ? -1 : 1, 0);
            return true;
        }
    } else {
        inside = Clouds_get(a, (int) floor(ox * invSize + a.cloudOffset.x), (int) floor(oz * invSize + a.cloudOffset.z));
    }

    // Where the ray leaves the cloud layer through its top or bottom
    float tLayerExit = d.y > 0 ? (top - oy) / d.y : d.y < 0 ? (bottom - oy) / d.y : HUGE_VALF;
    float tExit = fmin(tLayerExit, maxT);

    // Walk the cloud cells the ray crosses (2D DDA in the horizontal plane)
    float px = (ox + d.x * tStart) * invSize + a.cloudOffset.x;
    float pz = (oz + d.z * tStart) * invSize + a.cloudOffset.z;
    int ix = (int) floor(px);
    int iz = (int) floor(pz);
    int stepX = d.x > 0 ? 1 : -1;
    int stepZ = d.z > 0 ? 1 : -1;
    float dtx = d.x != 0 ? fabs(a.cloudSize / d.x) : HUGE_VALF;
    float dtz = d.z != 0 ? fabs(a.cloudSize / d.z) : HUGE_VALF;
    float tx = d.x > 0 ? (ix + 1 - px) * dtx : d.x < 0 ? (px - ix) * dtx : HUGE_VALF;
    float tz = d.z > 0 ? (iz + 1 - pz) * dtz : d.z < 0 ? (pz - iz) * dtz : HUGE_VALF;
    tx += tStart;
    tz += tStart;
    for (int i = 0; i < 4096; i++) {
        float tCell;
        float3 n;
        if (tx < tz) {
            tCell = tx;
            ix += stepX;
            tx += dtx;
            n = make_float3(-stepX, 0, 0);
        } else {
            tCell = tz;
            iz += stepZ;
            tz += dtz;
            n = make_float3(0, 0, -stepZ);
        }
        if (tCell >= tExit) {
            break;
        }
        if (Clouds_get(a, ix, iz) != inside) {
            *t = tCell;
            *normal = inside ? -n : n;
            return true;
        }
    }
    if (inside && tLayerExit <= maxT) {
        *t = tLayerExit;
        *normal = make_float3(0, d.y > 0 ? 1 : -1, 0);
        return true;
    }
    return false;
}

#endif
