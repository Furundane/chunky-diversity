#ifndef CHUNKYCL_TEXTURE_ATLAS_H
#define CHUNKYCL_TEXTURE_ATLAS_H

#include "../opencl.h"
#include "compat.h"
#include "constants.h"

#ifndef __METAL_VERSION__
// Image sampler for texture atlases.
const sampler_t Atlas_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#endif

float4 Atlas_read_xy(int x, int y, int location, image2d_array_t atlas) {
    x += ((location >> 22) & 0x1FF) * 16;
    y += ((location >> 13) & 0x1FF) * 16;
    // Location layout: x (9 bits) << 22 | y (9 bits) << 13 | layer (13 bits), see AtlasTexture.setLocation.
    // The layer mask used to be 0x7FFFF, which mixed y into the layer; OpenCL clamps the layer index so
    // that went unnoticed with a single-layer atlas.
    int d = location & 0x1FFF;

#ifdef __METAL_VERSION__
    // Callers clamp x and y to the texture (Atlas_read_uv), so an unsampled read matches the clamping sampler.
    return atlas.read(uint2(x, y), uint(d));
#else
    return read_imagef(atlas, Atlas_sampler, make_int4(x, y, d, 0));
#endif
}

float4 Atlas_read_uv(float u, float v, int location, int size, image2d_array_t atlas) {
    int width = (size >> 16) & 0xFFFF;
    int height = size & 0xFFFF;

    v = (1 - v);

    int x = clamp((int) ((u - EPS) * width), 0, width-1);
    int y = clamp((int) ((v - EPS) * height), 0, height-1);

    return Atlas_read_xy(x, y, location, atlas);
}

#endif
