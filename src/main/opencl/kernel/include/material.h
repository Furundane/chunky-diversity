// This includes stuff regarding materials

#ifndef CHUNKYCLPLUGIN_MATERIAL_H
#define CHUNKYCLPLUGIN_MATERIAL_H

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "textureAtlas.h"
#include "utils.h"
#include "constants.h"
#include "random.h"

typedef struct {
    __global const int* palette;
} MaterialPalette;

MaterialPalette MaterialPalette_new(__global const int* palette) {
    MaterialPalette p;
    p.palette = palette;
    return p;
}

typedef struct {
    unsigned int flags;
    unsigned int tint;
    unsigned int textureSize;
    unsigned int color;
    unsigned int normal_emittance;
    unsigned int specular_metalness_roughness;
    float ior;
} Material;

#define MATERIAL_REFRACTIVE 0b1000
#define DEFAULT_IOR 1.000293f  // Air, see Material.DEFAULT_IOR in Chunky

Material Material_get(MaterialPalette self, int material) {
    Material m;
    m.flags = self.palette[material + 0];
    m.tint = self.palette[material + 1];
    m.textureSize = self.palette[material + 2];
    m.color = self.palette[material + 3];
    m.normal_emittance = self.palette[material + 4];
    m.specular_metalness_roughness = self.palette[material + 5];
    m.ior = as_float(self.palette[material + 6]);
    return m;
}

typedef struct {
    float4 color;
    float emittance;
    float specular;
    float metalness;
    float roughness;
} MaterialSample;

bool Material_sample(Material self, image2d_array_t atlas, float2 uv, __private MaterialSample* sample) {
    // Color
    float4 color;
    if (self.flags & 0b100)
        color = Atlas_read_uv(uv.x, uv.y, self.color, self.textureSize, atlas);
    else
        color = colorFromArgb(self.color);
    
    if (self.tint == 0xFE000000) {
        // Light block
        sample->color.xyz = 0.5;
        sample->color.w = 1.0;
    } else if (color.w > EPS) {
        sample->color = color;
    } else {
        return false;
    }

    // Tint
    switch (self.tint >> 24) {
        case 0xFF:
            sample->color *= colorFromArgb(self.tint);
            break;
        case 1:
            // TODO Proper foliage tint
            sample->color *= colorFromArgb(0xFF71A74D);
            break;
        case 2:
            // TODO Proper grass tint
            sample->color *= colorFromArgb(0xFF8EB971);
            break;
        case 3:
            // TODO Proper water tint
            sample->color *= colorFromArgb(0xFF3F76E4);
            break;
    }

    // (Normal) emittance
    if (self.flags & 0b010)
        sample->emittance = Atlas_read_uv(uv.x, uv.y, self.normal_emittance, self.textureSize, atlas).w;
    else
        sample->emittance = (self.normal_emittance & 0xFF) / 255.0;

    // specular, metalness, roughness
    if (self.flags & 0b001) {
        float3 smr = Atlas_read_uv(uv.x, uv.y, self.specular_metalness_roughness, self.textureSize, atlas).xyz;
        sample->specular = smr.x;
        sample->metalness = smr.y;
        sample->roughness = smr.z;
    } else {
        sample->specular = (self.specular_metalness_roughness & 0xFF) / 255.0;
        sample->metalness = ((self.specular_metalness_roughness >> 8) & 0xFF) / 255.0;
        sample->roughness = ((self.specular_metalness_roughness >> 16) & 0xFF) / 255.0;
    }
    
    return true;
}

float3 _Material_diffuseReflection(IntersectionRecord record, Random random) {
    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);
    float r = sqrt(x1);
    float theta = 2 * M_PI_F * x2;

    float tx = r * cos(theta);
    float ty = r * sin(theta);
    float tz = sqrt(1 - x1);

    // Transform from tangent space to world space
    float xx, xy, xz;
    float ux, uy, uz;
    float vx, vy, vz;

    if (fabs(record.normal.x) > 0.1) {
        xx = 0;
        xy = 1;
    } else {
        xx = 1;
        xy = 0;
    }
    xz = 0;

    ux = xy * record.normal.z - xz * record.normal.y;
    uy = xz * record.normal.x - xx * record.normal.z;
    uz = xx * record.normal.y - xy * record.normal.x;

    r = 1 / sqrt(ux*ux + uy*uy + uz*uz);

    ux *= r;
    uy *= r;
    uz *= r;

    vx = uy * record.normal.z - uz * record.normal.y;
    vy = uz * record.normal.x - ux * record.normal.z;
    vz = ux * record.normal.y - uy * record.normal.x;

    return make_float3(
        ux * tx + vx * ty + record.normal.x * tz,
        uy * tx + vy * ty + record.normal.y * tz,
        uz * tx + vz * ty + record.normal.z * tz
    );
}

float3 _Material_specularReflection(IntersectionRecord record, MaterialSample sample, Ray ray, Random random) {
    float3 direction = ray.direction + (record.normal * (-2 * dot(ray.direction, record.normal)));

    if (sample.roughness > 0) {
        float3 diffuseDirection = _Material_diffuseReflection(record, random);
        diffuseDirection *= sample.roughness;
        direction = diffuseDirection + direction * (1 - sample.roughness);
    }

    if (signbit(dot(record.normal, direction)) == signbit(dot(record.normal, ray.direction))) {
        float factor = copysign(dot(record.normal, ray.direction), -EPS - dot(record.normal, direction));
        direction += factor * record.normal;
    }

    return normalize(direction);
}

typedef struct {
    float3 direction;
    float3 spectrum;
    bool specular;
    bool transmitted;  // The ray passed through the surface into the material
} MaterialPdfSample;

// PathTracer.reassignTransmissivity
float _Material_reassign(float from, float to, float other, float trans, float cap) {
    return (cap * (other - 2 * to + from) + (3 * trans) * (to - from)) / (other + to - 2 * from);
}

// Color of light transmitted through a translucent surface, see PathTracer.translucentRayColor. With fancier
// translucency, channels above the transmissivity cap give their excess to the other channels. The energy
// normalization Chunky does for caps above 1 needs the color of the next ray and is not done here.
float3 Material_transmission(float3 color, float absorption, bool fancier, float cap) {
    if (!fancier) {
        return (1 - absorption) + absorption * color;
    }
    float shouldTrans = 1 - absorption;
    float avg = (color.x + color.y + color.z) / 3;
    float3 t = avg > 0 ? color * (shouldTrans / avg) : (float3) (shouldTrans);
    float maxTrans = fmax(t.x, fmax(t.y, t.z));
    if (maxTrans > cap) {
        if (maxTrans == t.x) {
            float g = _Material_reassign(t.x, t.y, t.z, shouldTrans, cap);
            t.z = _Material_reassign(t.x, t.z, t.y, shouldTrans, cap);
            t.y = g;
            t.x = cap;
        } else if (maxTrans == t.y) {
            float r = _Material_reassign(t.y, t.x, t.z, shouldTrans, cap);
            t.z = _Material_reassign(t.y, t.z, t.x, shouldTrans, cap);
            t.x = r;
            t.y = cap;
        } else {
            float g = _Material_reassign(t.z, t.y, t.x, shouldTrans, cap);
            t.x = _Material_reassign(t.z, t.x, t.y, shouldTrans, cap);
            t.y = g;
            t.z = cap;
        }
    }
    return t;
}

// Scatter a ray at a surface, following PathTracer.pathTrace: metal or specular reflection, diffuse
// reflection with the texture alpha as probability, otherwise Fresnel reflection or transmission
// (refracted for refractive materials) tinted by the surface color. n1 is the index of refraction of
// the medium the ray comes from.
MaterialPdfSample Material_samplePdf(Material self, IntersectionRecord record, MaterialSample sample, Ray ray,
                                     float n1, bool fancier, float cap, Random random) {
    MaterialPdfSample out;
    out.transmitted = false;
    out.specular = true;

    // Face the incoming ray (quads are two-sided)
    IntersectionRecord facing = record;
    if (dot(facing.normal, ray.direction) > 0) {
        facing.normal = -facing.normal;
    }

    bool doMetal = sample.metalness > EPS && Random_nextFloat(random) < sample.metalness;
    if (doMetal || (sample.specular > EPS && Random_nextFloat(random) < sample.specular)) {
        out.direction = _Material_specularReflection(facing, sample, ray, random);
        out.spectrum = doMetal ? sample.color.xyz : (float3) (1);
        return out;
    }

    float alpha = sample.color.w;
    float3 color = sample.color.xyz;
    float pDiffuse = fancier ? 1 - pow(1 - alpha, fmax(color.x, fmax(color.y, color.z))) : alpha;
    if (Random_nextFloat(random) < pDiffuse) {
        out.direction = _Material_diffuseReflection(facing, random);
        out.spectrum = color;
        out.specular = false;
        return out;
    }
    float pAbsorb = fancier ? 1 - (1 - alpha) / (1 - pDiffuse + EPS) : alpha;

    float3 d = ray.direction;
    float3 n = facing.normal;
    out.direction = d;
    float n2 = self.ior;
    if (n1 != n2) {
        bool refractive = (self.flags & MATERIAL_REFRACTIVE) != 0;
        float n1n2 = n1 / n2;
        float cosTheta = -dot(n, d);
        float radicand = 1 - n1n2 * n1n2 * (1 - cosTheta * cosTheta);
        if (refractive && radicand < EPS) {
            // Total internal reflection
            out.direction = _Material_specularReflection(facing, sample, ray, random);
            out.spectrum = (float3) (1);
            return out;
        }
        float a = n1n2 - 1;
        float b = n1n2 + 1;
        float r0 = a * a / (b * b);
        float c = 1 - cosTheta;
        float fresnel = r0 + (1 - r0) * c * c * c * c * c;
        if (Random_nextFloat(random) < fresnel) {
            out.direction = _Material_specularReflection(facing, sample, ray, random);
            out.spectrum = (float3) (1);
            return out;
        }
        if (refractive) {
            out.direction = normalize(n1n2 * d + (n1n2 * cosTheta - sqrt(radicand)) * n);
        }
    }
    out.spectrum = Material_transmission(color, pAbsorb, fancier, cap);
    out.transmitted = true;
    return out;
}

#endif
