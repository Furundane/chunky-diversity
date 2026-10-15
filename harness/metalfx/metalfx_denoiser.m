// Experiment: MetalFX temporal denoiser (macOS 26) on frames from the GPU path tracer. Separate from the
// plugin's Metal bridge; inputs come from Java as float arrays and are uploaded each frame (not timed).
//   harness/metalfx/build.sh   builds build/libmetalfxdenoiser.dylib
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <jni.h>

@interface MfxDenoiser : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLFXTemporalDenoisedScaler> scaler;
@property int width, height;
@property id<MTLTexture> color, depth, motion, normal, diffuse, specular, roughness, output, exposure;
@property id<MTLBuffer> colorStage, depthStage, motionStage, normalStage, diffuseStage, specularStage, roughnessStage, outputStage, exposureStage;
@end
@implementation MfxDenoiser
@end

static id<MTLTexture> newTexture(MfxDenoiser *d, MTLPixelFormat format, MTLTextureUsage usage, int w, int h) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    desc.usage = usage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [d.device newTextureWithDescriptor:desc];
}

static void fail(const char *message) {
    fprintf(stderr, "metalfx: %s\n", message);
}

JNIEXPORT jlong JNICALL Java_harness_MetalFxDenoiser_create(JNIEnv *env, jclass cls, jint width, jint height) {
    @autoreleasepool {
        MfxDenoiser *d = [MfxDenoiser new];
        d.device = MTLCreateSystemDefaultDevice();
        d.queue = [d.device newCommandQueue];
        d.width = width;
        d.height = height;
        if (@available(macOS 26.0, *)) {
            if (![MTLFXTemporalDenoisedScalerDescriptor supportsDevice:d.device]) {
                fail("denoised scaler not supported on this device");
                return 0;
            }
            MTLFXTemporalDenoisedScalerDescriptor *desc = [MTLFXTemporalDenoisedScalerDescriptor new];
            desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
            desc.depthTextureFormat = MTLPixelFormatDepth32Float;
            desc.motionTextureFormat = MTLPixelFormatRG16Float;
            desc.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
            desc.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
            desc.normalTextureFormat = MTLPixelFormatRGBA16Float;
            desc.roughnessTextureFormat = MTLPixelFormatR16Float;
            desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
            desc.inputWidth = width;
            desc.inputHeight = height;
            desc.outputWidth = width;
            desc.outputHeight = height;
            desc.requiresSynchronousInitialization = YES;
            desc.autoExposureEnabled = NO;
            d.scaler = [desc newTemporalDenoisedScalerWithDevice:d.device];
            if (d.scaler == nil) {
                fail("could not create the denoised scaler");
                return 0;
            }
            id<MTLFXTemporalDenoisedScaler> s = d.scaler;
            d.color = newTexture(d, MTLPixelFormatRGBA16Float, s.colorTextureUsage, width, height);
            d.depth = newTexture(d, MTLPixelFormatDepth32Float, s.depthTextureUsage, width, height);
            d.motion = newTexture(d, MTLPixelFormatRG16Float, s.motionTextureUsage, width, height);
            d.diffuse = newTexture(d, MTLPixelFormatRGBA16Float, s.diffuseAlbedoTextureUsage, width, height);
            d.specular = newTexture(d, MTLPixelFormatRGBA16Float, s.specularAlbedoTextureUsage, width, height);
            d.normal = newTexture(d, MTLPixelFormatRGBA16Float, s.normalTextureUsage, width, height);
            d.roughness = newTexture(d, MTLPixelFormatR16Float, s.roughnessTextureUsage, width, height);
            d.output = newTexture(d, MTLPixelFormatRGBA16Float, s.outputTextureUsage, width, height);
            d.exposure = newTexture(d, MTLPixelFormatR16Float, MTLTextureUsageShaderRead, 1, 1);
            long px = (long) width * height;
            MTLResourceOptions shared = MTLResourceStorageModeShared;
            d.colorStage = [d.device newBufferWithLength:px * 8 options:shared];
            d.depthStage = [d.device newBufferWithLength:px * 4 options:shared];
            d.motionStage = [d.device newBufferWithLength:px * 4 options:shared];
            d.diffuseStage = [d.device newBufferWithLength:px * 8 options:shared];
            d.specularStage = [d.device newBufferWithLength:px * 8 options:shared];
            d.normalStage = [d.device newBufferWithLength:px * 8 options:shared];
            d.roughnessStage = [d.device newBufferWithLength:px * 2 options:shared];
            d.outputStage = [d.device newBufferWithLength:px * 8 options:shared];
            d.exposureStage = [d.device newBufferWithLength:16 options:shared];
            ((__fp16 *) d.exposureStage.contents)[0] = 1.0f;
            return (jlong) CFBridgingRetain(d);
        } else {
            fail("needs macOS 26");
            return 0;
        }
    }
}

// Packs `channels` floats per pixel into `outChannels` halfs per pixel (missing channels are 0).
static void packHalf(JNIEnv *env, jfloatArray array, int channels, int outChannels, id<MTLBuffer> stage, long px) {
    jfloat *src = (*env)->GetPrimitiveArrayCritical(env, array, NULL);
    __fp16 *dst = stage.contents;
    for (long i = 0; i < px; i++) {
        for (int c = 0; c < outChannels; c++) {
            dst[i * outChannels + c] = c < channels ? (__fp16) src[i * channels + c] : (__fp16) 0;
        }
    }
    (*env)->ReleasePrimitiveArrayCritical(env, array, src, JNI_ABORT);
}

static void upload(id<MTLBlitCommandEncoder> blit, id<MTLBuffer> stage, id<MTLTexture> texture, int bytesPerPixel) {
    [blit copyFromBuffer:stage sourceOffset:0 sourceBytesPerRow:texture.width * bytesPerPixel
         sourceBytesPerImage:texture.width * texture.height * bytesPerPixel
                  sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                   toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
}

static simd_float4x4 matrix(const float *m) {
    simd_float4x4 r;
    for (int c = 0; c < 4; c++) r.columns[c] = simd_make_float4(m[c * 4], m[c * 4 + 1], m[c * 4 + 2], m[c * 4 + 3]);
    return r;
}

// color: rgb, depth: device depth, motion: xy in pixels, normal/diffuse/specular: xyz, roughness: 1 per pixel.
// params: jitter x, jitter y, reset history (0/1), world-to-view (16, column-major), view-to-clip (16).
// Writes rgb to out and returns the GPU time of the denoiser in nanoseconds (-1 on error).
JNIEXPORT jlong JNICALL Java_harness_MetalFxDenoiser_denoise(JNIEnv *env, jclass cls, jlong handle,
        jfloatArray color, jfloatArray depth, jfloatArray motion, jfloatArray normal, jfloatArray diffuse,
        jfloatArray specular, jfloatArray roughness, jfloatArray params, jfloatArray out) {
    @autoreleasepool {
        if (@available(macOS 26.0, *)) {
            MfxDenoiser *d = (__bridge MfxDenoiser *) (void *) handle;
            long px = (long) d.width * d.height;
            packHalf(env, color, 3, 4, d.colorStage, px);
            packHalf(env, motion, 2, 2, d.motionStage, px);
            packHalf(env, normal, 3, 4, d.normalStage, px);
            packHalf(env, diffuse, 3, 4, d.diffuseStage, px);
            packHalf(env, specular, 3, 4, d.specularStage, px);
            packHalf(env, roughness, 1, 1, d.roughnessStage, px);
            jfloat *dsrc = (*env)->GetPrimitiveArrayCritical(env, depth, NULL);
            memcpy(d.depthStage.contents, dsrc, px * 4);
            (*env)->ReleasePrimitiveArrayCritical(env, depth, dsrc, JNI_ABORT);
            float p[35];
            (*env)->GetFloatArrayRegion(env, params, 0, 35, p);

            id<MTLCommandBuffer> uploadBuffer = [d.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [uploadBuffer blitCommandEncoder];
            upload(blit, d.colorStage, d.color, 8);
            upload(blit, d.depthStage, d.depth, 4);
            upload(blit, d.motionStage, d.motion, 4);
            upload(blit, d.normalStage, d.normal, 8);
            upload(blit, d.diffuseStage, d.diffuse, 8);
            upload(blit, d.specularStage, d.specular, 8);
            upload(blit, d.roughnessStage, d.roughness, 2);
            upload(blit, d.exposureStage, d.exposure, 2);
            [blit endEncoding];
            [uploadBuffer commit];

            id<MTLFXTemporalDenoisedScaler> s = d.scaler;
            s.colorTexture = d.color;
            s.depthTexture = d.depth;
            s.motionTexture = d.motion;
            s.normalTexture = d.normal;
            s.diffuseAlbedoTexture = d.diffuse;
            s.specularAlbedoTexture = d.specular;
            s.roughnessTexture = d.roughness;
            s.outputTexture = d.output;
            s.exposureTexture = d.exposure;
            s.jitterOffsetX = p[0];
            s.jitterOffsetY = p[1];
            s.shouldResetHistory = p[2] != 0;
            s.motionVectorScaleX = 1;
            s.motionVectorScaleY = 1;
            s.depthReversed = NO;
            s.worldToViewMatrix = matrix(p + 3);
            s.viewToClipMatrix = matrix(p + 19);
            id<MTLCommandBuffer> denoiseBuffer = [d.queue commandBuffer];
            [s encodeToCommandBuffer:denoiseBuffer];
            [denoiseBuffer commit];

            id<MTLCommandBuffer> readBuffer = [d.queue commandBuffer];
            blit = [readBuffer blitCommandEncoder];
            [blit copyFromTexture:d.output sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(d.width, d.height, 1) toBuffer:d.outputStage destinationOffset:0
          destinationBytesPerRow:d.width * 8 destinationBytesPerImage:px * 8];
            [blit endEncoding];
            [readBuffer commit];
            [readBuffer waitUntilCompleted];
            if (denoiseBuffer.error != nil || readBuffer.error != nil || uploadBuffer.error != nil) {
                NSError *e = denoiseBuffer.error ?: (uploadBuffer.error ?: readBuffer.error);
                fprintf(stderr, "metalfx: %s\n", e.localizedDescription.UTF8String);
                return -1;
            }

            jfloat *dst = (*env)->GetPrimitiveArrayCritical(env, out, NULL);
            const __fp16 *o = d.outputStage.contents;
            for (long i = 0; i < px; i++) {
                dst[i * 3] = o[i * 4];
                dst[i * 3 + 1] = o[i * 4 + 1];
                dst[i * 3 + 2] = o[i * 4 + 2];
            }
            (*env)->ReleasePrimitiveArrayCritical(env, out, dst, 0);
            return (jlong) ((denoiseBuffer.GPUEndTime - denoiseBuffer.GPUStartTime) * 1e9);
        }
        return -1;
    }
}

JNIEXPORT void JNICALL Java_harness_MetalFxDenoiser_release(JNIEnv *env, jclass cls, jlong handle) {
    if (handle != 0) CFBridgingRelease((void *) handle);
}
