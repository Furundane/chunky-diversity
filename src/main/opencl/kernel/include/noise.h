// 3D simplex noise with analytic derivatives, as se.llbit.math.SimplexNoise in Chunky (Stefan Gustavson's
// sdnoise1234, public domain, via Keijiro Takahashi's Processing port). Used for water waves.

#ifndef CHUNKYCL_NOISE_H
#define CHUNKYCL_NOISE_H

#include "../opencl.h"
#include "compat.h"

__constant int SIMPLEX_PERM[512] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
    37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
    77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
    164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
    221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
    49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13,
    201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87,
    174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
    209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17,
    182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180
};

__constant float SIMPLEX_GRAD3[16][3] = {{1, 0, 1}, {0, 1, 1}, {-1, 0, 1}, {0, -1, 1}, {1, 0, -1}, {0, 1, -1}, {-1, 0, -1}, {0, -1, -1}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {0, 1, -1}, {0, -1, -1}};

// Noise contribution of one simplex corner, and its two parts of the derivative (see SimplexNoise.calculate).
void _Simplex_corner(int gi, float x, float y, float z, __private float* n, __private float3* t3dot, __private float3* t4g) {
    float t = 0.6f - x * x - y * y - z * z;
    if (t < 0) {
        *n = 0;
        *t3dot = make_float3(0, 0, 0);
        *t4g = make_float3(0, 0, 0);
        return;
    }
    float gx = SIMPLEX_GRAD3[gi & 15][0];
    float gy = SIMPLEX_GRAD3[gi & 15][1];
    float gz = SIMPLEX_GRAD3[gi & 15][2];
    float t2 = t * t;
    float t4 = t2 * t2;
    float dot = gx * x + gy * y + gz * z;
    *n = t4 * dot;
    *t3dot = make_float3(x, y, z) * (t2 * t * dot);
    *t4g = make_float3(gx, gy, gz) * t4;
}

// 3D simplex noise; also returns the derivatives with respect to x and y.
float Simplex_noise3(float x, float y, float z, __private float* ddx, __private float* ddy) {
    const float F3 = 1.0f / 3;
    const float G3 = 1.0f / 6;
    float s = (x + y + z) * F3;
    int i = (int) floor(x + s);
    int j = (int) floor(y + s);
    int k = (int) floor(z + s);
    float t = (i + j + k) * G3;
    float x0 = x - (i - t);
    float y0 = y - (j - t);
    float z0 = z - (k - t);

    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
        else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
        else               { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
        if (y0 < z0)       { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
        else if (x0 < z0)  { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
        else               { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }
    float x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
    float x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;

    float n0, n1, n2, n3;
    float3 a0, a1, a2, a3, b0, b1, b2, b3;
    _Simplex_corner(SIMPLEX_PERM[ii + SIMPLEX_PERM[jj + SIMPLEX_PERM[kk]]], x0, y0, z0, &n0, &a0, &b0);
    _Simplex_corner(SIMPLEX_PERM[ii + i1 + SIMPLEX_PERM[jj + j1 + SIMPLEX_PERM[kk + k1]]], x1, y1, z1, &n1, &a1, &b1);
    _Simplex_corner(SIMPLEX_PERM[ii + i2 + SIMPLEX_PERM[jj + j2 + SIMPLEX_PERM[kk + k2]]], x2, y2, z2, &n2, &a2, &b2);
    _Simplex_corner(SIMPLEX_PERM[ii + 1 + SIMPLEX_PERM[jj + 1 + SIMPLEX_PERM[kk + 1]]], x3, y3, z3, &n3, &a3, &b3);

    float3 d = ((a0 + a1 + a2 + a3) * -8 + b0 + b1 + b2 + b3) * 28;
    *ddx = d.x;
    *ddy = d.y;
    return 28 * (n0 + n1 + n2 + n3);
}

#endif
