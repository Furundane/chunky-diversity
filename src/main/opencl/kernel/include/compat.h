// Portability layer. The kernel is written in OpenCL C; this header lets the same source also
// compile as Metal Shading Language (detected by __METAL_VERSION__). It only covers the
// constructs the kernel actually uses.
//
// Rules for kernel code:
//  - Vector literals with more than one argument use make_float3(a, b, c) etc., not
//    (float3)(a, b, c). In C++ (Metal) the latter is a cast of a comma expression and silently
//    becomes float3(c). Single-argument splats like (float3)(0) are fine in both.
//  - Pointers to function-local data are declared __private (Metal requires an address space).

#ifndef CHUNKYCL_COMPAT_H
#define CHUNKYCL_COMPAT_H

#ifdef __METAL_VERSION__

#include <metal_stdlib>
using namespace metal;

// Address spaces
#define __global device
#define __private thread
#define __kernel kernel
#define __constant constant

// Images. The texture read is written per backend where it occurs (textureAtlas.h).
#define image2d_array_t texture2d_array<float>

// Reinterpreting casts
#define as_float(x) as_type<float>(x)
#define as_int(x) as_type<int>(x)
#define as_uint(x) as_type<uint>(x)

// Vector literals
#define make_float2(...) float2(__VA_ARGS__)
#define make_float3(...) float3(__VA_ARGS__)
#define make_float4(...) float4(__VA_ARGS__)
#define make_int3(...) int3(__VA_ARGS__)
#define make_int4(...) int4(__VA_ARGS__)

// Kernel entry points end their parameter list with KERNEL_GID_PARAM and read GLOBAL_ID
// instead of get_global_id(0).
#define KERNEL_GID_PARAM , uint chunkycl_gid [[thread_position_in_grid]]
#define GLOBAL_ID ((int) chunkycl_gid)

// Packed 3-component loads and stores (float3 is 16 bytes in Metal, as in OpenCL)
inline float3 vload3(size_t i, const device float* p) {
    return float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
}
inline float3 vload3(size_t i, const thread float* p) {
    return float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
}
inline void vstore3(float3 v, size_t i, device float* p) {
    p[3 * i] = v.x;
    p[3 * i + 1] = v.y;
    p[3 * i + 2] = v.z;
}

#else

#define KERNEL_GID_PARAM
#define GLOBAL_ID get_global_id(0)


// Some OpenCL headers (Apple's) predefine fixed-arity versions; these accept any arity,
// e.g. make_float3(someFloat2, z).
#undef make_float2
#undef make_float3
#undef make_float4
#undef make_int3
#undef make_int4
#define make_float2(...) ((float2)(__VA_ARGS__))
#define make_float3(...) ((float3)(__VA_ARGS__))
#define make_float4(...) ((float4)(__VA_ARGS__))
#define make_int3(...) ((int3)(__VA_ARGS__))
#define make_int4(...) ((int4)(__VA_ARGS__))

#endif

#endif
