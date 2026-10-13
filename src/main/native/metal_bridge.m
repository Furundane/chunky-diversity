// JNI side of dev.thatredox.chunkynative.metal.Metal. Built with ARC (-fobjc-arc); handles given to
// Java are CFBridgingRetain'd objects and are freed by Metal.release.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <jni.h>
#include <string.h>
#include "dev_thatredox_chunkynative_metal_Metal.h"

@interface CMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@end

@implementation CMContext
@end

static void throwRuntime(JNIEnv *env, NSString *message) {
    jclass cls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, cls, message.UTF8String);
}

static jlong toHandle(id object) {
    return (jlong) (intptr_t) CFBridgingRetain(object);
}

static id fromHandle(jlong handle) {
    return (__bridge id) (void *) (intptr_t) handle;
}

static NSString *toNSString(JNIEnv *env, jstring string) {
    const char *chars = (*env)->GetStringUTFChars(env, string, NULL);
    NSString *result = [NSString stringWithUTF8String:chars];
    (*env)->ReleaseStringUTFChars(env, string, chars);
    return result;
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_createContext(JNIEnv *env, jclass cls) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throwRuntime(env, @"No Metal device");
            return 0;
        }
        CMContext *context = [CMContext new];
        context.device = device;
        context.queue = [device newCommandQueue];
        return toHandle(context);
    }
}

JNIEXPORT jstring JNICALL Java_dev_thatredox_chunkynative_metal_Metal_deviceName(JNIEnv *env, jclass cls, jlong contextHandle) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        return (*env)->NewStringUTF(env, context.device.name.UTF8String);
    }
}

static id<MTLLibrary> compileLibrary(JNIEnv *env, CMContext *context, jstring source, jboolean fastMath) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = fastMath ? MTLMathModeFast : MTLMathModeSafe;
    } else {
        options.fastMathEnabled = fastMath;
    }

    NSError *error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:toNSString(env, source) options:options error:&error];
    if (library == nil) {
        throwRuntime(env, [@"Metal compile failed:\n" stringByAppendingString:error.localizedDescription]);
        return nil;
    }
    if (error != nil) {
        // A library with an error object means the compile succeeded with warnings.
        fprintf(stderr, "Metal compile warnings:\n%s\n", error.localizedDescription.UTF8String);
    }
    return library;
}

static jlong newPipeline(JNIEnv *env, CMContext *context, id<MTLLibrary> library, NSString *functionName) {
    id<MTLFunction> kernel = [library newFunctionWithName:functionName];
    if (kernel == nil) {
        throwRuntime(env, [NSString stringWithFormat:@"No kernel function named %@", functionName]);
        return 0;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [context.device newComputePipelineStateWithFunction:kernel error:&error];
    if (pipeline == nil) {
        throwRuntime(env, [@"Pipeline creation failed: " stringByAppendingString:error.localizedDescription]);
        return 0;
    }
    return toHandle(pipeline);
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compile(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jstring function, jboolean fastMath) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return 0;
        return newPipeline(env, context, library, toNSString(env, function));
    }
}

JNIEXPORT jlongArray JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compileFunctions(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jobjectArray functions, jboolean fastMath) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return NULL;
        jsize count = (*env)->GetArrayLength(env, functions);
        jlong pipelines[count];
        for (jsize i = 0; i < count; i++) {
            jstring name = (*env)->GetObjectArrayElement(env, functions, i);
            pipelines[i] = newPipeline(env, context, library, toNSString(env, name));
            if (pipelines[i] == 0) {
                for (jsize k = 0; k < i; k++) CFBridgingRelease((void *) (intptr_t) pipelines[k]);
                return NULL;
            }
        }
        jlongArray result = (*env)->NewLongArray(env, count);
        (*env)->SetLongArrayRegion(env, result, 0, count, pipelines);
        return result;
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_newBuffer(JNIEnv *env, jclass cls, jlong contextHandle, jlong bytes) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLBuffer> buffer = [context.device newBufferWithLength:(NSUInteger) bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throwRuntime(env, [NSString stringWithFormat:@"Could not allocate a %lld byte buffer", (long long) bytes]);
            return 0;
        }
        memset(buffer.contents, 0, buffer.length);
        return toHandle(buffer);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_maxBufferLength(JNIEnv *env, jclass cls, jlong contextHandle) {
    CMContext *context = fromHandle(contextHandle);
    return (jlong) context.device.maxBufferLength;
}

// Pointer to bytes [byteOffset, byteOffset + bytes) of a buffer, or NULL (with an exception) if out of range.
static void *bufferRange(JNIEnv *env, jlong bufferHandle, jlong byteOffset, jlong bytes) {
    id<MTLBuffer> buffer = fromHandle(bufferHandle);
    if (byteOffset < 0 || bytes < 0 || (NSUInteger) (byteOffset + bytes) > buffer.length) {
        throwRuntime(env, [NSString stringWithFormat:@"Range %lld + %lld is outside a %lu byte buffer",
                (long long) byteOffset, (long long) bytes, (unsigned long) buffer.length]);
        return NULL;
    }
    return (char *) buffer.contents + byteOffset;
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_writeInts(JNIEnv *env, jclass cls, jlong bufferHandle,
        jlong byteOffset, jintArray data, jint offset, jint length) {
    void *target = bufferRange(env, bufferHandle, byteOffset, (jlong) length * 4);
    if (target != NULL) {
        (*env)->GetIntArrayRegion(env, data, offset, length, (jint *) target);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_writeFloats(JNIEnv *env, jclass cls, jlong bufferHandle,
        jlong byteOffset, jfloatArray data, jint offset, jint length) {
    void *target = bufferRange(env, bufferHandle, byteOffset, (jlong) length * 4);
    if (target != NULL) {
        (*env)->GetFloatArrayRegion(env, data, offset, length, (jfloat *) target);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_readInts(JNIEnv *env, jclass cls, jlong bufferHandle,
        jlong byteOffset, jintArray data, jint offset, jint length) {
    void *source = bufferRange(env, bufferHandle, byteOffset, (jlong) length * 4);
    if (source != NULL) {
        (*env)->SetIntArrayRegion(env, data, offset, length, (const jint *) source);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_readFloats(JNIEnv *env, jclass cls, jlong bufferHandle,
        jlong byteOffset, jfloatArray data, jint offset, jint length) {
    void *source = bufferRange(env, bufferHandle, byteOffset, (jlong) length * 4);
    if (source != NULL) {
        (*env)->SetFloatArrayRegion(env, data, offset, length, (const jfloat *) source);
    }
}

static jlong newTexture(JNIEnv *env, jlong contextHandle, MTLTextureType type, jint width, jint height, jint layers) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = type;
        descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
        descriptor.width = (NSUInteger) width;
        descriptor.height = (NSUInteger) height;
        descriptor.arrayLength = (NSUInteger) layers;
        descriptor.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [context.device newTextureWithDescriptor:descriptor];
        if (texture == nil) {
            throwRuntime(env, [NSString stringWithFormat:@"Could not allocate a %dx%dx%d texture", width, height, layers]);
            return 0;
        }
        return toHandle(texture);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_newTexture2DArray(JNIEnv *env, jclass cls, jlong contextHandle,
        jint width, jint height, jint layers) {
    return newTexture(env, contextHandle, MTLTextureType2DArray, width, height, layers);
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_newTexture2D(JNIEnv *env, jclass cls, jlong contextHandle,
        jint width, jint height) {
    return newTexture(env, contextHandle, MTLTextureType2D, width, height, 1);
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_replaceRegion(JNIEnv *env, jclass cls, jlong textureHandle,
        jint x, jint y, jint layer, jint width, jint height, jbyteArray rgba) {
    id<MTLTexture> texture = fromHandle(textureHandle);
    if ((*env)->GetArrayLength(env, rgba) < (jsize) width * height * 4) {
        throwRuntime(env, @"Pixel array is smaller than the region");
        return;
    }
    jbyte *pixels = (*env)->GetPrimitiveArrayCritical(env, rgba, NULL);
    [texture replaceRegion:MTLRegionMake2D((NSUInteger) x, (NSUInteger) y, (NSUInteger) width, (NSUInteger) height)
               mipmapLevel:0
                     slice:(NSUInteger) layer
                 withBytes:pixels
               bytesPerRow:(NSUInteger) width * 4
             bytesPerImage:0];
    (*env)->ReleasePrimitiveArrayCritical(env, rgba, pixels, JNI_ABORT);
}

// Binds buffers and textures at indices 0..n-1, in the order the kernel declares them, and encodes the dispatch.
static void encodeDispatch(JNIEnv *env, id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                           jlongArray bufferHandles, jlongArray textureHandles, jint threads) {
    [encoder setComputePipelineState:pipeline];

    jsize count = (*env)->GetArrayLength(env, bufferHandles);
    jlong *handles = (*env)->GetLongArrayElements(env, bufferHandles, NULL);
    for (jsize i = 0; i < count; i++) {
        [encoder setBuffer:fromHandle(handles[i]) offset:0 atIndex:(NSUInteger) i];
    }
    (*env)->ReleaseLongArrayElements(env, bufferHandles, handles, JNI_ABORT);

    count = (*env)->GetArrayLength(env, textureHandles);
    handles = (*env)->GetLongArrayElements(env, textureHandles, NULL);
    for (jsize i = 0; i < count; i++) {
        [encoder setTexture:fromHandle(handles[i]) atIndex:(NSUInteger) i];
    }
    (*env)->ReleaseLongArrayElements(env, textureHandles, handles, JNI_ABORT);

    // Apple GPUs support non-uniform threadgroups, so the grid need not be a multiple of the group size.
    NSUInteger groupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) threads);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger) threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
}

// Waits for a command buffer; returns its GPU time in ns, or 0 with an exception if it failed.
static jlong finish(JNIEnv *env, id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
        throwRuntime(env, [@"Kernel failed: " stringByAppendingString:commandBuffer.error.localizedDescription]);
        return 0;
    }
    return (jlong) ((commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1e9);
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_dispatch(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint threads) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        if (threads <= 0) {
            return 0;
        }
        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        encodeDispatch(env, encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, threads);
        [encoder endEncoding];
        return finish(env, commandBuffer);
    }
}

// A batch of dispatches in one command buffer. The compute encoder is serial: each dispatch sees
// the results of the ones before it.
@interface CMBatch : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> commandBuffer;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
@end

@implementation CMBatch
@end

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_beginBatch(JNIEnv *env, jclass cls, jlong contextHandle) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        CMBatch *batch = [CMBatch new];
        batch.commandBuffer = [context.queue commandBuffer];
        batch.encoder = [batch.commandBuffer computeCommandEncoder];
        return toHandle(batch);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_addDispatch(JNIEnv *env, jclass cls, jlong batchHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint threads) {
    @autoreleasepool {
        CMBatch *batch = fromHandle(batchHandle);
        if (threads > 0) {
            encodeDispatch(env, batch.encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, threads);
        }
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_runBatch(JNIEnv *env, jclass cls, jlong batchHandle) {
    @autoreleasepool {
        CMBatch *batch = CFBridgingRelease((void *) (intptr_t) batchHandle);
        [batch.encoder endEncoding];
        return finish(env, batch.commandBuffer);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_release(JNIEnv *env, jclass cls, jlong handle) {
    if (handle != 0) {
        CFBridgingRelease((void *) (intptr_t) handle);
    }
}
