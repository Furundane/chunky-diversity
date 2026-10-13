#ifndef CHUNKYCLPLUGIN_CAMERA_H
#define CHUNKYCLPLUGIN_CAMERA_H

#include "../opencl.h"
#include "compat.h"
#include "rt.h"
#include "random.h"

Ray Camera_preGenerated(__global const float* rays, int index) {
    Ray ray;
    ray.origin = vload3(index * 2, rays);
    ray.direction = vload3(index * 2 + 1, rays);
    return ray;
}

Ray Camera_pinHole(float x, float y, Random random, __global const float* projectorSettings) {
    Ray ray;
    float aperature = projectorSettings[0];
    float subjectDistance = projectorSettings[1];
    float fovTan = projectorSettings[2];

    ray.origin = make_float3(0, 0, 0);
    ray.direction = make_float3(fovTan * x, fovTan * y, 1.0);

    if (aperature > 0) {
        ray.direction *= subjectDistance / ray.direction.z;

        float r = sqrt(Random_nextFloat(random)) * aperature;
        float theta = Random_nextFloat(random) * M_PI_F * 2.0;
        float rx = cos(theta) * r;
        float ry = sin(theta) * r;

        ray.direction -= make_float3(rx, ry, 0);
        ray.origin += make_float3(rx, ry, 0);
    }

    return ray;
}

// Camera ray for pixel gid: from the projector settings, or pre-generated (see CameraData).
Ray ray_to_camera(
        const __global int* projectorType,
        const __global float* cameraSettings,
        const __global int* canvasConfig,
        int gid,
        Random random
) {
    Ray ray;
    if (*projectorType != -1) {
        float3 cameraPos = vload3(0, cameraSettings);
        float3 m1s = vload3(1, cameraSettings);
        float3 m2s = vload3(2, cameraSettings);
        float3 m3s = vload3(3, cameraSettings);

        int width = canvasConfig[0];
        int fullWidth = canvasConfig[2];
        int fullHeight = canvasConfig[3];
        int cropX = canvasConfig[4];
        int cropY = canvasConfig[5];

        float halfWidth = fullWidth / (2.0 * fullHeight);
        float invHeight = 1.0 / fullHeight;
        float x = -halfWidth + ((gid % width) + Random_nextFloat(random) + cropX) * invHeight;
        float y = -0.5 + ((gid / width) + Random_nextFloat(random) + cropY) * invHeight;

        switch (*projectorType) {
            case 0:
                ray = Camera_pinHole(x, y, random, cameraSettings + 12);
                break;
        }

        ray.direction = normalize(make_float3(
                dot(m1s, ray.direction),
                        dot(m2s, ray.direction),
                        dot(m3s, ray.direction)
        ));
        ray.origin = make_float3(
                dot(m1s, ray.origin),
                        dot(m2s, ray.origin),
                        dot(m3s, ray.origin)
        );

        ray.origin += cameraPos;
    } else {
        ray = Camera_preGenerated(cameraSettings, gid);
    }
    return ray;
}

#endif
