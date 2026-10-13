#ifndef CHUNKYCLPLUGIN_SKY_H
#define CHUNKYCLPLUGIN_SKY_H

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "textureAtlas.h"
#include "random.h"
#include "material.h"

typedef struct {
    int flags;
    int textureSize;
    int texture;
    float luminosity;
    float3 su;
    float3 sv;
    float3 sw;
    float3 color;                       // Sun color, for the sun seen by diffuse rays
    float3 apparentTextureBrightness;   // For the sun texture seen directly
    float radius;
    float3 emittance;      // Sun.getEmittance(): color * intensity ^ gamma, used for direct light
    float luminosityPdf;
} Sun;

// How the sun is added when a ray escapes to the sky
#define SUN_NONE    0  // Not at all (the sun is sampled as direct light instead)
#define SUN_DIFFUSE 1  // As seen by diffuse reflections (Sun.intersectDiffuse)
#define SUN_TEXTURE 2  // The sun texture (camera rays and specular reflections)

Sun Sun_new(__global const int* data) {
    Sun sun;
    sun.flags = data[0];
    sun.textureSize = data[1];
    sun.texture = data[2];
    sun.luminosity = as_float(data[5]);
    sun.radius = as_float(data[6]);
    sun.color = make_float3(as_float(data[7]), as_float(data[8]), as_float(data[9]));
    sun.apparentTextureBrightness = make_float3(as_float(data[10]), as_float(data[11]), as_float(data[12]));
    sun.emittance = make_float3(as_float(data[13]), as_float(data[14]), as_float(data[15]));
    sun.luminosityPdf = as_float(data[16]);

    float phi = as_float(data[3]);
    float theta = as_float(data[4]);
    float r = fabs(cos(phi));
    
    sun.sw = make_float3(cos(theta) * r, sin(phi), sin(theta) * r);
    if (fabs(sun.sw.x) > 0.1f) {
        sun.su = make_float3(0, 1, 0);
    } else {
        sun.su = make_float3(1, 0, 0);
    }
    sun.sv = normalize(cross(sun.sw, sun.su));
    sun.su = cross(sun.sv, sun.sw);
    
    return sun;
}

bool Sun_intersect(Sun self, image2d_array_t atlas, Ray ray, int mode, __private MaterialSample* sample) {
    float3 direction = ray.direction;

    // The sun texture is only drawn for rays that see the sun directly (Sun.intersect); diffuse rays see
    // it regardless (Sun.intersectDiffuse).
    if (mode == SUN_NONE || (mode == SUN_TEXTURE && !(self.flags & 1)) || dot(direction, self.sw) < 0.5f) {
        return false;
    }

    float width = self.radius * 4;
    float width2 = width * 2;
    float a = M_PI_2_F - acos(dot(direction, self.su)) + width;
    if (a >= 0 && a < width2) {
        float b = M_PI_2_F - acos(dot(direction, self.sv)) + width;
        if (b >= 0 && b < width2) {
            float3 texture = Atlas_read_uv(a / width2, b / width2, self.texture, self.textureSize, atlas).xyz;
            if (mode == SUN_DIFFUSE) {
                // Sky.addSunColorDiffuseSun: the sun intensity does not apply here
                sample->color.xyz += texture * self.color * 10 * self.luminosity;
            } else {
                // Sky.addSunColor
                sample->color.xyz += texture * self.apparentTextureBrightness * 10;
            }
            return true;
        }
    }

    return false;
}

// Random direction towards the sun disk, see Sun.getRandomSunDirection.
float3 Sun_sampleDirection(Sun self, Random random) {
    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);

    float cos_a = 1 - x1 + x1 * cos(self.radius);
    float sin_a = sqrt(fmax(0.0f, 1 - cos_a * cos_a));
    float phi = 2 * M_PI_F * x2;

    return normalize(self.su * (cos(phi) * sin_a) + self.sv * (sin(phi) * sin_a) + self.sw * cos_a);
}

// The baked sky (SkyData): the resolution, then linear RGB floats per texel, row y = elevation.
float3 _Sky_texel(__global const float* sky, int resolution, int x, int y) {
    return vload3(y * resolution + x, sky + 1);
}

void Sky_intersect(__global const float* sky, float skyMultiplier, Ray ray, __private MaterialSample* sample) {
    float3 direction = ray.direction;

    float theta = atan2(direction.z, direction.x);
    theta /= M_PI_F * 2;
    theta = fmod(fmod(theta, 1) + 1, 1);
    float phi = (asin(clamp(direction.y, -1.0f, 1.0f)) + M_PI_2_F) * M_1_PI_F;
    // Straight up or down, atan2(0, 0) is NaN on some GPUs (Metal); an index computed from it would read
    // outside the buffer. The azimuth does not matter there.
    if (!(theta >= 0 && theta < 1)) theta = 0;

    // Bilinear interpolation. Texel (x, y) holds the sky at azimuth x / resolution and elevation
    // y / resolution; the azimuth wraps around and the elevation is clamped at the poles.
    int resolution = (int) sky[0];
    float fx = theta * resolution;
    float fy = phi * resolution;
    int x0 = (int) floor(fx);
    int y0 = (int) floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    int x1 = (x0 + 1) % resolution;
    x0 = x0 % resolution;
    int ya = clamp(y0, 0, resolution - 1);
    int yb = clamp(y0 + 1, 0, resolution - 1);
    float3 bottom = mix(_Sky_texel(sky, resolution, x0, ya), _Sky_texel(sky, resolution, x1, ya), tx);
    float3 top = mix(_Sky_texel(sky, resolution, x0, yb), _Sky_texel(sky, resolution, x1, yb), tx);

    sample->color = make_float4(mix(bottom, top, ty) * skyMultiplier, 1);
    sample->emittance = 1.0;
}

#endif
