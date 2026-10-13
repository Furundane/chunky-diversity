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
    float intensity;
    float luminosity;
    float3 su;
    float3 sv;
    float3 sw;
    float4 color;
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
    sun.intensity = as_float(data[3]);
    sun.luminosity = as_float(data[6]);
    sun.color = colorFromArgb(data[7]);
    sun.radius = as_float(data[8]);
    sun.emittance = make_float3(as_float(data[9]), as_float(data[10]), as_float(data[11]));
    sun.luminosityPdf = as_float(data[12]);
    
    float phi = as_float(data[4]);
    float theta = as_float(data[5]);
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

    if (mode == SUN_NONE || !(self.flags & 1) || dot(direction, self.sw) < 0.5f) {
        return false;
    }

    float width = self.radius * 4;
    float width2 = width * 2;
    float a = M_PI_2_F - acos(dot(direction, self.su)) + width;
    if (a >= 0 && a < width2) {
        float b = M_PI_2_F - acos(dot(direction, self.sv)) + width;
        if (b >= 0 && b < width2) {
            if (mode == SUN_DIFFUSE) {
                float4 color = self.color;
                color *= self.luminosity;
                sample->color += color;
            } else {
                float4 color = Atlas_read_uv(a / width2, b / width2,
                                             self.texture, self.textureSize, atlas);
                color *= self.intensity;
                sample->color += color;
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

#ifdef __METAL_VERSION__
constexpr sampler skySampler(coord::normalized, address::mirrored_repeat, filter::linear);
#define Sky_read(texture, uv) (texture).sample(skySampler, (uv))
#else
const sampler_t skySampler = CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_LINEAR;
#define Sky_read(texture, uv) read_imagef((texture), skySampler, (uv))
#endif

void Sky_intersect(image2d_t skyTexture, float skyIntensity, Ray ray, __private MaterialSample* sample) {
    float3 direction = ray.direction;

    float theta = atan2(direction.z, direction.x);
    theta /= M_PI_F * 2;
    theta = fmod(fmod(theta, 1) + 1, 1);
    float phi = (asin(clamp(direction.y, -1.0f, 1.0f)) + M_PI_2_F) * M_1_PI_F;

    sample->color = Sky_read(skyTexture, make_float2(theta, phi)) * skyIntensity;
    sample->emittance = 1.0;
}

#endif
