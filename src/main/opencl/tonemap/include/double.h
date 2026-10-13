#ifndef KERNEL_DOUBLE_H
#define KERNEL_DOUBLE_H

#include "../opencl.h"

// DEVICE_FP64 comes from the host, which checks the device's extension list. The cl_khr_fp64
// macro alone is not enough: Apple's compiler defines it for GPUs that cannot run doubles.
#ifdef DEVICE_FP64
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define CL_DOUBLE_SUPPORT
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define CL_DOUBLE_SUPPORT
#endif
#endif

typedef ulong imposter_double;

/// Convert an (imposter) double to a float.
float idouble_to_float(imposter_double value) {
#ifdef CL_DOUBLE_SUPPORT
    return (float) as_double(value);
#else
    // Builds the float's bits directly (mantissa truncated to 23 bits). Converting the ulong
    // mantissa to float instead crashes Apple's OpenCL-to-Metal compiler.
    uint sign = (uint) (value >> 63) << 31;
    int exponent = (int) ((value >> 52) & 0x7FF);
    ulong mantissa = value & ((1ull << 52) - 1);

    if (exponent == 0x7FF) {
        // NAN or INFINITY
        return mantissa == 0 ? as_float(sign | 0x7F800000u) : NAN;
    }
    int floatExponent = exponent - 1023 + 127;
    if (floatExponent >= 0xFF) {
        // Too large for a float
        return as_float(sign | 0x7F800000u);
    }
    if (floatExponent <= 0) {
        // Zero, or too small for a normal float
        return as_float(sign);
    }
    return as_float(sign | ((uint) floatExponent << 23) | (uint) (mantissa >> 29));
#endif
}

#endif
