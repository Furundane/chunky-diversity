// JNI side of dev.thatredox.chunkynative.metal.Metal. Built with ARC (-fobjc-arc); handles given to
// Java are CFBridgingRetain'd objects and are freed by Metal.release.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <jni.h>
#include <string.h>
#include "dev_thatredox_chunkynative_metal_Metal.h"

@interface CMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
// More queues for batches that run at the same time (beginBatchOn), created on first use.
@property(nonatomic, strong) NSMutableArray<id<MTLCommandQueue>> *moreQueues;
@end

@implementation CMContext
@end

// A compute pipeline linked with an intersection function (hardware ray tracing, see hwrt.h).
@interface CMRtPipeline : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic, strong) id<MTLFunction> intersectionFunction;
@end

@implementation CMRtPipeline
@end

// The world octree as bounding boxes: acceleration structure plus the box data the intersection
// function reads.
@interface CMBlockAccel : NSObject
@property(nonatomic, strong) id<MTLAccelerationStructure> structure;
@property(nonatomic, strong) id<MTLBuffer> boxes;
@property(nonatomic, strong) id<MTLBuffer> info;
@property(nonatomic, strong) id<MTLBuffer> bitmaps;
@property(nonatomic, strong) id<MTLBuffer> params;
@property(nonatomic) jlong boxCount;
@property(nonatomic) jlong buildNanos;
@end

@implementation CMBlockAccel
@end

// An intersection function table with its buffers bound, and every resource the function reads
// through it (these must be made resident with useResource when dispatching).
@interface CMFunctionTable : NSObject
@property(nonatomic, strong) id<MTLIntersectionFunctionTable> table;
@property(nonatomic, strong) NSArray<id<MTLResource>> *resources;
@end

@implementation CMFunctionTable
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
        // Experiment: CHUNKYMETAL_MATH=relaxed|fast overrides the math mode, CHUNKYMETAL_FUNCTIONS=fast the math functions.
        const char *math = getenv("CHUNKYMETAL_MATH");
        if (math != NULL && strcmp(math, "relaxed") == 0) options.mathMode = MTLMathModeRelaxed;
        if (math != NULL && strcmp(math, "fast") == 0) options.mathMode = MTLMathModeFast;
        const char *functions = getenv("CHUNKYMETAL_FUNCTIONS");
        if (functions != NULL && strcmp(functions, "fast") == 0) options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
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

// A pipeline for a kernel function; with an intersection function name, a CMRtPipeline linked with it.
static jlong newPipeline(JNIEnv *env, CMContext *context, id<MTLLibrary> library, NSString *functionName,
                         NSString *intersectionName) {
    id<MTLFunction> kernel = [library newFunctionWithName:functionName];
    if (kernel == nil) {
        throwRuntime(env, [NSString stringWithFormat:@"No kernel function named %@", functionName]);
        return 0;
    }
    NSError *error = nil;
    if (intersectionName == nil) {
        id<MTLComputePipelineState> pipeline = [context.device newComputePipelineStateWithFunction:kernel error:&error];
        if (pipeline == nil) {
            throwRuntime(env, [@"Pipeline creation failed: " stringByAppendingString:error.localizedDescription]);
            return 0;
        }
        return toHandle(pipeline);
    }
    id<MTLFunction> intersection = [library newFunctionWithName:intersectionName];
    if (intersection == nil) {
        throwRuntime(env, [NSString stringWithFormat:@"No intersection function named %@", intersectionName]);
        return 0;
    }
    MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
    linked.functions = @[intersection];
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = kernel;
    descriptor.linkedFunctions = linked;
    id<MTLComputePipelineState> pipeline = [context.device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
    if (pipeline == nil) {
        throwRuntime(env, [@"Pipeline creation failed: " stringByAppendingString:error.localizedDescription]);
        return 0;
    }
    CMRtPipeline *result = [CMRtPipeline new];
    result.pipeline = pipeline;
    result.intersectionFunction = intersection;
    return toHandle(result);
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compile(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jstring function, jboolean fastMath) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return 0;
        return newPipeline(env, context, library, toNSString(env, function), nil);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compileRt(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jstring function, jboolean fastMath, jstring intersectionFunction) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return 0;
        return newPipeline(env, context, library, toNSString(env, function), toNSString(env, intersectionFunction));
    }
}

// Pipelines for several kernel functions of one source; with intersectionFunction (may be null),
// CMRtPipelines linked with it.
JNIEXPORT jlongArray JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compileFunctions(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jobjectArray functions, jboolean fastMath, jstring intersectionFunction) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return NULL;
        NSString *intersection = intersectionFunction != NULL ? toNSString(env, intersectionFunction) : nil;
        jsize count = (*env)->GetArrayLength(env, functions);
        jlong pipelines[count];
        for (jsize i = 0; i < count; i++) {
            jstring name = (*env)->GetObjectArrayElement(env, functions, i);
            pipelines[i] = newPipeline(env, context, library, toNSString(env, name), intersection);
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
// Buffer handles may also be block acceleration structures and intersection function tables.
// indirect: dispatch the number of threadgroups (of `threads` threads) stored in that buffer at
// indirectOffset (MTLDispatchThreadgroupsIndirectArguments), instead of `threads` threads.
static void encodeDispatch(JNIEnv *env, id<MTLComputeCommandEncoder> encoder, id pipelineObject,
                           jlongArray bufferHandles, jlongArray textureHandles, jint threads,
                           id<MTLBuffer> indirect, jlong indirectOffset) {
    id<MTLComputePipelineState> pipeline = [pipelineObject isKindOfClass:[CMRtPipeline class]]
            ? ((CMRtPipeline *) pipelineObject).pipeline : pipelineObject;
    // Make the writes of earlier dispatches in this encoder (path state, queues, and the indirect
    // dispatch arguments written by wf_prepare) visible to this one.
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:pipeline];

    jsize count = (*env)->GetArrayLength(env, bufferHandles);
    jlong *handles = (*env)->GetLongArrayElements(env, bufferHandles, NULL);
    for (jsize i = 0; i < count; i++) {
        id object = fromHandle(handles[i]);
        if ([object isKindOfClass:[CMBlockAccel class]]) {
            CMBlockAccel *accel = object;
            [encoder setAccelerationStructure:accel.structure atBufferIndex:(NSUInteger) i];
            [encoder useResource:accel.structure usage:MTLResourceUsageRead];
        } else if ([object isKindOfClass:[CMFunctionTable class]]) {
            CMFunctionTable *table = object;
            [encoder setIntersectionFunctionTable:table.table atBufferIndex:(NSUInteger) i];
            for (id<MTLResource> resource in table.resources) {
                [encoder useResource:resource usage:MTLResourceUsageRead];
            }
        } else {
            [encoder setBuffer:object offset:0 atIndex:(NSUInteger) i];
        }
    }
    (*env)->ReleaseLongArrayElements(env, bufferHandles, handles, JNI_ABORT);

    count = (*env)->GetArrayLength(env, textureHandles);
    handles = (*env)->GetLongArrayElements(env, textureHandles, NULL);
    for (jsize i = 0; i < count; i++) {
        [encoder setTexture:fromHandle(handles[i]) atIndex:(NSUInteger) i];
    }
    (*env)->ReleaseLongArrayElements(env, textureHandles, handles, JNI_ABORT);

    if (indirect != nil) {
        [encoder dispatchThreadgroupsWithIndirectBuffer:indirect
                                   indirectBufferOffset:(NSUInteger) indirectOffset
                                  threadsPerThreadgroup:MTLSizeMake((NSUInteger) threads, 1, 1)];
        return;
    }
    // Apple GPUs support non-uniform threadgroups, so the grid need not be a multiple of the group size.
    NSUInteger groupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) threads);
    // Experiment: CHUNKYMETAL_GROUP=n caps the threadgroup size.
    const char *group = getenv("CHUNKYMETAL_GROUP");
    if (group != NULL && atoi(group) > 0) groupSize = MIN(groupSize, (NSUInteger) atoi(group));
    [encoder dispatchThreads:MTLSizeMake((NSUInteger) threads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
}

// Diagnostics: with MTL_CAPTURE_ENABLED=1 and CHUNKYMETAL_CAPTURE=<file.gputrace>, the
// CHUNKYMETAL_CAPTURE_DISPATCH-th command buffer (default 10) is captured for Xcode.
static bool startCapture(id<MTLCommandQueue> queue) {
    static int commandBuffers = 0;
    const char *capturePath = getenv("CHUNKYMETAL_CAPTURE");
    if (capturePath == NULL) return false;
    const char *which = getenv("CHUNKYMETAL_CAPTURE_DISPATCH");
    if (++commandBuffers != (which != NULL ? atoi(which) : 10)) return false;
    MTLCaptureDescriptor *descriptor = [MTLCaptureDescriptor new];
    descriptor.captureObject = queue;
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@(capturePath)];
    NSError *captureError = nil;
    bool started = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&captureError];
    fprintf(stderr, "Metal capture to %s: %s\n", capturePath,
            started ? "started" : captureError.localizedDescription.UTF8String);
    return started;
}

// Waits for a command buffer; returns its GPU time in ns, or 0 with an exception if it failed.
static jlong finish(JNIEnv *env, id<MTLCommandBuffer> commandBuffer, bool capture) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (capture) {
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
        fprintf(stderr, "Metal capture saved\n");
    }
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
        bool capture = startCapture(context.queue);
        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        encodeDispatch(env, encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, threads, nil, 0);
        [encoder endEncoding];
        return finish(env, commandBuffer, capture);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_dispatchIndirect(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint groupSize,
        jlong indirectHandle, jlong indirectOffset) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        bool capture = startCapture(context.queue);
        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        encodeDispatch(env, encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, groupSize,
                       fromHandle(indirectHandle), indirectOffset);
        [encoder endEncoding];
        return finish(env, commandBuffer, capture);
    }
}

// A batch of dispatches in one command buffer. The compute encoder is serial: each dispatch sees
// the results of the ones before it.
@interface CMBatch : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> commandBuffer;
@property(nonatomic, strong) id<MTLComputeCommandEncoder> encoder;
@property(nonatomic) bool capture;
@end

@implementation CMBatch
@end

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_beginBatch(JNIEnv *env, jclass cls, jlong contextHandle) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        CMBatch *batch = [CMBatch new];
        batch.capture = startCapture(context.queue);
        batch.commandBuffer = [context.queue commandBuffer];
        batch.encoder = [batch.commandBuffer computeCommandEncoder];
        return toHandle(batch);
    }
}

// A batch on command queue `queue` (0: the context's queue). Batches on different queues may run
// at the same time on the GPU (see runBatches).
JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_beginBatchOn(JNIEnv *env, jclass cls, jlong contextHandle, jint queue) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLCommandQueue> commandQueue = context.queue;
        if (queue > 0) {
            if (context.moreQueues == nil) context.moreQueues = [NSMutableArray array];
            while ((jint) context.moreQueues.count < queue) [context.moreQueues addObject:[context.device newCommandQueue]];
            commandQueue = context.moreQueues[(NSUInteger) queue - 1];
        }
        CMBatch *batch = [CMBatch new];
        batch.capture = false;
        batch.commandBuffer = [commandQueue commandBuffer];
        batch.encoder = [batch.commandBuffer computeCommandEncoder];
        return toHandle(batch);
    }
}

// Commits several batches together, waits for all of them and frees them. Returns the GPU time
// from the first start to the last end in nanoseconds, or 0 with an exception if one failed.
JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_runBatches(JNIEnv *env, jclass cls, jlongArray batchHandles) {
    @autoreleasepool {
        jsize count = (*env)->GetArrayLength(env, batchHandles);
        jlong *handles = (*env)->GetLongArrayElements(env, batchHandles, NULL);
        NSMutableArray<CMBatch *> *batches = [NSMutableArray array];
        for (jsize i = 0; i < count; i++) {
            CMBatch *batch = CFBridgingRelease((void *) (intptr_t) handles[i]);
            [batch.encoder endEncoding];
            [batch.commandBuffer commit];
            [batches addObject:batch];
        }
        (*env)->ReleaseLongArrayElements(env, batchHandles, handles, JNI_ABORT);
        CFTimeInterval start = INFINITY, end = 0;
        NSString *failure = nil;
        for (CMBatch *batch in batches) {
            [batch.commandBuffer waitUntilCompleted];
            if (batch.commandBuffer.status == MTLCommandBufferStatusError) {
                failure = batch.commandBuffer.error.localizedDescription;
            }
            start = MIN(start, batch.commandBuffer.GPUStartTime);
            end = MAX(end, batch.commandBuffer.GPUEndTime);
        }
        if (failure != nil) {
            throwRuntime(env, [@"Kernel failed: " stringByAppendingString:failure]);
            return 0;
        }
        if (getenv("CHUNKYMETAL_BATCH_TIMES") != NULL) {
            // Diagnostic: when each command buffer ran on the GPU, relative to the first start (ms).
            NSMutableString *line = [NSMutableString stringWithString:@"BATCH_TIMES"];
            for (CMBatch *batch in batches) {
                [line appendFormat:@" [%.2f..%.2f]", (batch.commandBuffer.GPUStartTime - start) * 1e3,
                        (batch.commandBuffer.GPUEndTime - start) * 1e3];
            }
            fprintf(stderr, "%s\n", line.UTF8String);
        }
        return count == 0 ? 0 : (jlong) ((end - start) * 1e9);
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_addDispatch(JNIEnv *env, jclass cls, jlong batchHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint threads) {
    @autoreleasepool {
        CMBatch *batch = fromHandle(batchHandle);
        if (threads > 0) {
            encodeDispatch(env, batch.encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, threads, nil, 0);
        }
    }
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_addDispatchIndirect(JNIEnv *env, jclass cls, jlong batchHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint groupSize,
        jlong indirectHandle, jlong indirectOffset) {
    @autoreleasepool {
        CMBatch *batch = fromHandle(batchHandle);
        encodeDispatch(env, batch.encoder, fromHandle(pipelineHandle), bufferHandles, textureHandles, groupSize,
                       fromHandle(indirectHandle), indirectOffset);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_runBatch(JNIEnv *env, jclass cls, jlong batchHandle) {
    @autoreleasepool {
        CMBatch *batch = CFBridgingRelease((void *) (intptr_t) batchHandle);
        [batch.encoder endEncoding];
        return finish(env, batch.commandBuffer, batch.capture);
    }
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_retain(JNIEnv *env, jclass cls, jlong handle) {
    if (handle != 0) {
        CFRetain((CFTypeRef) (intptr_t) handle);
    }
    return handle;
}

JNIEXPORT void JNICALL Java_dev_thatredox_chunkynative_metal_Metal_release(JNIEnv *env, jclass cls, jlong handle) {
    if (handle != 0) {
        CFBridgingRelease((void *) (intptr_t) handle);
    }
}

// ---------------------------------------------------------------------------------------------------
// Hardware ray tracing of the world octree (kernel side: hwrt.h).

JNIEXPORT jboolean JNICALL Java_dev_thatredox_chunkynative_metal_Metal_supportsRaytracing(JNIEnv *env, jclass cls, jlong contextHandle) {
    CMContext *context = fromHandle(contextHandle);
    return context.device.supportsRaytracing ? JNI_TRUE : JNI_FALSE;
}

typedef struct {
    float *boxes;  // 6 per box
    int *info;     // 4 per box
    uint32_t *bitmaps;
    size_t count, capacity, words, wordCapacity;
} BoxList;

static void boxListAdd(BoxList *list, int x, int y, int z, int sx, int sy, int sz, const int lo[3], int w) {
    if (list->count == list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 1024;
        list->boxes = realloc(list->boxes, list->capacity * 6 * sizeof(float));
        list->info = realloc(list->info, list->capacity * 4 * sizeof(int));
    }
    float *b = list->boxes + list->count * 6;
    b[0] = (float) (x + lo[0]); b[1] = (float) (y + lo[1]); b[2] = (float) (z + lo[2]);
    b[3] = (float) (x + sx); b[4] = (float) (y + sy); b[5] = (float) (z + sz);
    int *i = list->info + list->count * 4;
    i[0] = x; i[1] = y; i[2] = z; i[3] = w;
    list->count++;
}

typedef struct {
    const int *tree;
    const int *palette;
    size_t paletteLength;
    int shift;  // Section size 2^shift, 0 for leaves only
} OctreeSource;

// Can a ray hit this packed block? Air, ANY_TYPE and invisible blocks (model type 0) never are.
static inline bool occupied(const OctreeSource *src, int data) {
    return data > 0 && data != 0x7FFFFFFE && (size_t) data < src->paletteLength && src->palette[data] != 0;
}

static void rasterize(const OctreeSource *src, int index, int x, int y, int z, int level, int ox, int oy, int oz,
                      uint32_t *bits, int lo[3], int hi[3], bool *any) {
    int v = src->tree[index];
    if (v > 0) {
        int half = 1 << (level - 1);
        for (int c = 0; c < 8; c++) {
            rasterize(src, v + c, x + ((c >> 2) & 1) * half, y + ((c >> 1) & 1) * half, z + (c & 1) * half, level - 1,
                      ox, oy, oz, bits, lo, hi, any);
        }
        return;
    }
    if (!occupied(src, -v)) return;
    int size = 1 << level, s = src->shift;
    for (int bx = x; bx < x + size; bx++)
        for (int by = y; by < y + size; by++)
            for (int bz = z; bz < z + size; bz++) {
                int bit = ((bx - ox) << (2 * s)) | ((by - oy) << s) | (bz - oz);
                bits[bit >> 5] |= 1u << (bit & 31);
            }
    lo[0] = MIN(lo[0], x - ox); lo[1] = MIN(lo[1], y - oy); lo[2] = MIN(lo[2], z - oz);
    hi[0] = MAX(hi[0], x - ox + size); hi[1] = MAX(hi[1], y - oy + size); hi[2] = MAX(hi[2], z - oz + size);
    *any = true;
}

// Leaves of one block become one box each (of any size); with sections, the blocks below a node of
// the section size are collected into one box with an occupancy bitmap.
static void collectBoxes(const OctreeSource *src, int index, int x, int y, int z, int level, BoxList *out) {
    int v = src->tree[index];
    if (v <= 0) {
        if (occupied(src, -v)) {
            int size = 1 << level;
            const int zero[3] = {0, 0, 0};
            boxListAdd(out, x, y, z, size, size, size, zero, -1 - (-v));
        }
        return;
    }
    if (src->shift > 0 && level == src->shift) {
        int words = MAX(1, (1 << (3 * src->shift)) / 32);
        uint32_t bits[128] = {0};
        int lo[3] = {1 << level, 1 << level, 1 << level}, hi[3] = {0, 0, 0};
        bool any = false;
        rasterize(src, index, x, y, z, level, x, y, z, bits, lo, hi, &any);
        if (!any) return;
        if (out->words + words > out->wordCapacity) {
            out->wordCapacity = MAX(out->wordCapacity * 2, out->words + words + 1024);
            out->bitmaps = realloc(out->bitmaps, out->wordCapacity * sizeof(uint32_t));
        }
        memcpy(out->bitmaps + out->words, bits, words * sizeof(uint32_t));
        // The box is the tight bound of the occupied blocks; info keeps the section origin.
        boxListAdd(out, x, y, z, hi[0], hi[1], hi[2], lo, (int) out->words);
        out->words += words;
        return;
    }
    int half = 1 << (level - 1);
    for (int c = 0; c < 8; c++) {
        collectBoxes(src, v + c, x + ((c >> 2) & 1) * half, y + ((c >> 1) & 1) * half, z + (c & 1) * half, level - 1, out);
    }
}

typedef struct {
    int index, x, y, z, level;
} Subtree;

static void collectSubtrees(const OctreeSource *src, Subtree node, int stopLevel, Subtree **list, size_t *count, size_t *capacity) {
    if (node.level <= stopLevel || src->tree[node.index] <= 0) {
        if (*count == *capacity) {
            *capacity = *capacity ? *capacity * 2 : 256;
            *list = realloc(*list, *capacity * sizeof(Subtree));
        }
        (*list)[(*count)++] = node;
        return;
    }
    int v = src->tree[node.index], half = 1 << (node.level - 1);
    for (int c = 0; c < 8; c++) {
        Subtree child = {v + c, node.x + ((c >> 2) & 1) * half, node.y + ((c >> 1) & 1) * half, node.z + (c & 1) * half, node.level - 1};
        collectSubtrees(src, child, stopLevel, list, count, capacity);
    }
}

/*
 * Builds the boxes and the acceleration structure for the (packed) world octree. sectionShift: 2, 3
 * or 4 for 4^3, 8^3 or 16^3 sections, 0 for one box per octree leaf. Throws when the structure would
 * need more than budgetBytes (0: half of the GPU's recommended working set minus what is already
 * allocated).
 */
JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_buildBlockAccel(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong octreeHandle, jint depth, jlong paletteHandle, jint sectionShift, jlong budgetBytes) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLDevice> device = context.device;
        id<MTLBuffer> octree = fromHandle(octreeHandle);
        id<MTLBuffer> palette = fromHandle(paletteHandle);
        if (!device.supportsRaytracing) {
            throwRuntime(env, @"This GPU does not support Metal ray tracing");
            return 0;
        }
        uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);

        OctreeSource src = {(const int *) octree.contents, (const int *) palette.contents, palette.length / 4, sectionShift};
        Subtree *roots = NULL;
        size_t rootCount = 0, rootCapacity = 0;
        Subtree root = {0, 0, 0, 0, depth};
        collectSubtrees(&src, root, MAX(sectionShift, depth - 4), &roots, &rootCount, &rootCapacity);
        BoxList *parts = calloc(rootCount, sizeof(BoxList));
        const OctreeSource *srcPtr = &src;
        const Subtree *rootsPtr = roots;
        dispatch_apply(rootCount, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            collectBoxes(srcPtr, rootsPtr[i].index, rootsPtr[i].x, rootsPtr[i].y, rootsPtr[i].z, rootsPtr[i].level, &parts[i]);
        });
        size_t boxCount = 0, wordCount = 0;
        for (size_t i = 0; i < rootCount; i++) {
            boxCount += parts[i].count;
            wordCount += parts[i].words;
        }

        NSUInteger budget = (NSUInteger) budgetBytes;
        if (budget == 0) {
            NSUInteger half = (NSUInteger) (device.recommendedMaxWorkingSetSize / 2);
            budget = half > device.currentAllocatedSize ? half - device.currentAllocatedSize : 0;
        }
        id<MTLBuffer> boxes = nil, info = nil, bitmaps = nil;
        size_t dataBytes = boxCount * 40 + wordCount * 4;
        if (boxCount > 0 && dataBytes < budget) {
            boxes = [device newBufferWithLength:MAX(boxCount * 24, 16) options:MTLResourceStorageModeShared];
            info = [device newBufferWithLength:MAX(boxCount * 16, 16) options:MTLResourceStorageModeShared];
            bitmaps = [device newBufferWithLength:MAX(wordCount * 4, 16) options:MTLResourceStorageModeShared];
        }
        if (boxes != nil && info != nil && bitmaps != nil) {
            size_t box = 0, word = 0;
            for (size_t i = 0; i < rootCount; i++) {
                BoxList *part = &parts[i];
                // Bitmap word offsets are relative to the part: shift them to the merged array.
                for (size_t k = 0; k < part->count; k++) {
                    if (part->info[k * 4 + 3] >= 0) part->info[k * 4 + 3] += (int) word;
                }
                memcpy((float *) boxes.contents + box * 6, part->boxes, part->count * 6 * sizeof(float));
                memcpy((int *) info.contents + box * 4, part->info, part->count * 4 * sizeof(int));
                if (part->words) memcpy((uint32_t *) bitmaps.contents + word, part->bitmaps, part->words * 4);
                box += part->count;
                word += part->words;
            }
        }
        for (size_t i = 0; i < rootCount; i++) {
            free(parts[i].boxes);
            free(parts[i].info);
            free(parts[i].bitmaps);
        }
        free(parts);
        free(roots);
        if (boxCount == 0) {
            throwRuntime(env, @"The world has no blocks");
            return 0;
        }
        if (boxes == nil || info == nil || bitmaps == nil) {
            throwRuntime(env, [NSString stringWithFormat:@"%zu boxes need %.2f GB, more than the %.2f GB available",
                    boxCount, dataBytes / 1e9, budget / 1e9]);
            return 0;
        }

        MTLAccelerationStructureBoundingBoxGeometryDescriptor *geometry = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        geometry.boundingBoxBuffer = boxes;
        geometry.boundingBoxStride = 24;
        geometry.boundingBoxCount = boxCount;
        geometry.intersectionFunctionTableOffset = 0;
        MTLPrimitiveAccelerationStructureDescriptor *descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        descriptor.geometryDescriptors = @[geometry];
        // Experiment: CHUNKYMETAL_AS_USAGE=fast|min (macOS 26 build preferences).
        if (@available(macOS 26.0, *)) {
            const char *usage = getenv("CHUNKYMETAL_AS_USAGE");
            if (usage != NULL && strcmp(usage, "fast") == 0) descriptor.usage = MTLAccelerationStructureUsagePreferFastIntersection;
            if (usage != NULL && strcmp(usage, "min") == 0) descriptor.usage = MTLAccelerationStructureUsageMinimizeMemory;
        }
        MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:descriptor];
        size_t need = dataBytes + sizes.accelerationStructureSize + sizes.buildScratchBufferSize;
        if (need > budget || sizes.accelerationStructureSize > device.maxBufferLength) {
            throwRuntime(env, [NSString stringWithFormat:@"%zu boxes need %.2f GB to build, more than the %.2f GB available",
                    boxCount, need / 1e9, budget / 1e9]);
            return 0;
        }
        id<MTLAccelerationStructure> full = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch = [device newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];
        id<MTLBuffer> compactedSize = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (full == nil || scratch == nil) {
            throwRuntime(env, @"Could not allocate the acceleration structure");
            return 0;
        }
        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> encoder = [commandBuffer accelerationStructureCommandEncoder];
        [encoder buildAccelerationStructure:full descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
        [encoder writeCompactedAccelerationStructureSize:full toBuffer:compactedSize offset:0];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            throwRuntime(env, [@"Acceleration structure build failed: " stringByAppendingString:commandBuffer.error.localizedDescription]);
            return 0;
        }
        scratch = nil;
        id<MTLAccelerationStructure> compact = [device newAccelerationStructureWithSize:*(uint32_t *) compactedSize.contents];
        commandBuffer = [context.queue commandBuffer];
        encoder = [commandBuffer accelerationStructureCommandEncoder];
        [encoder copyAndCompactAccelerationStructure:full toAccelerationStructure:compact];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        int params[2] = {sectionShift, depth};
        CMBlockAccel *accel = [CMBlockAccel new];
        accel.structure = compact;
        accel.boxes = boxes;
        accel.info = info;
        accel.bitmaps = bitmaps;
        accel.params = [device newBufferWithBytes:params length:sizeof(params) options:MTLResourceStorageModeShared];
        accel.boxCount = (jlong) boxCount;
        accel.buildNanos = (jlong) (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start);
        return toHandle(accel);
    }
}

/** {box count, acceleration structure bytes, box data bytes, build time in ns}. */
JNIEXPORT jlongArray JNICALL Java_dev_thatredox_chunkynative_metal_Metal_blockAccelInfo(JNIEnv *env, jclass cls, jlong accelHandle) {
    CMBlockAccel *accel = fromHandle(accelHandle);
    jlong values[4] = {accel.boxCount, (jlong) accel.structure.size,
                       (jlong) (accel.boxes.length + accel.info.length + accel.bitmaps.length), accel.buildNanos};
    jlongArray result = (*env)->NewLongArray(env, 4);
    (*env)->SetLongArrayRegion(env, result, 0, 4, values);
    return result;
}

/*
 * Intersection function table for an rt pipeline: the box data of accel, then sceneBuffers (octree,
 * block palette, quad models, AABB models, material palette), then the atlas texture through an
 * argument buffer, in the order HwRt_intersectBox declares them.
 */
JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_newFunctionTable(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong pipelineHandle, jlong accelHandle, jlongArray sceneBuffers, jlong atlasHandle) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        CMRtPipeline *pipeline = fromHandle(pipelineHandle);
        CMBlockAccel *accel = fromHandle(accelHandle);
        id<MTLTexture> atlas = fromHandle(atlasHandle);

        MTLIntersectionFunctionTableDescriptor *descriptor = [MTLIntersectionFunctionTableDescriptor new];
        descriptor.functionCount = 1;
        id<MTLIntersectionFunctionTable> table = [pipeline.pipeline newIntersectionFunctionTableWithDescriptor:descriptor];
        [table setFunction:[pipeline.pipeline functionHandleWithFunction:pipeline.intersectionFunction] atIndex:0];

        NSMutableArray<id<MTLResource>> *resources = [NSMutableArray array];
        NSArray<id<MTLBuffer>> *own = @[accel.boxes, accel.info, accel.bitmaps, accel.params];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer in own) {
            [table setBuffer:buffer offset:0 atIndex:index++];
            [resources addObject:buffer];
        }
        jsize count = (*env)->GetArrayLength(env, sceneBuffers);
        jlong *handles = (*env)->GetLongArrayElements(env, sceneBuffers, NULL);
        for (jsize i = 0; i < count; i++) {
            id<MTLBuffer> buffer = fromHandle(handles[i]);
            [table setBuffer:buffer offset:0 atIndex:index++];
            [resources addObject:buffer];
        }
        (*env)->ReleaseLongArrayElements(env, sceneBuffers, handles, JNI_ABORT);

        // The atlas as a resource ID (Metal 3 argument buffer layout of a struct with one texture).
        if (atlas != nil) {
            id<MTLBuffer> textures = nil;
            if (@available(macOS 13.0, *)) {
                MTLResourceID atlasId = atlas.gpuResourceID;
                textures = [context.device newBufferWithBytes:&atlasId length:sizeof(atlasId) options:MTLResourceStorageModeShared];
            } else {
                throwRuntime(env, @"Textures in intersection functions need macOS 13 or newer");
                return 0;
            }
            [table setBuffer:textures offset:0 atIndex:index++];
            [resources addObject:textures];
            [resources addObject:atlas];
        }

        CMFunctionTable *result = [CMFunctionTable new];
        result.table = table;
        result.resources = resources;
        return toHandle(result);
    }
}

// ---------------------------------------------------------------------------------------------------
// Denoised preview (MetalDenoisedPreviewRenderer): MetalFX temporal denoised scaler, macOS 26.
// The render kernel (CHUNKYCL_AOV) writes color and per-pixel inputs to buffers; denoise_pack turns
// them into the scaler's textures (with device depth and motion vectors from the camera of this and
// the previous frame), the scaler denoises and upscales, and denoise_unpack tone maps to ARGB.

static NSString *const denoiseSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"// p: [0..14] camera (position, transform rows, aperture, subject distance, fov tan), [15..29] previous camera,\n"
"// [30] input width, [31] input height, [32] full width, [33] full height, [34] crop x, [35] crop y (input pixels),\n"
"// [36] near, [37] far, [38] has previous camera, [39] exposure, [40] tone mapping, [41] output width, [42] output height\n"
"static float2 toPixel(constant float* cam, float3 d, bool sky, constant float* p, thread bool& ok) {\n"
"    if (!sky) d -= float3(cam[0], cam[1], cam[2]);\n"
"    float cx = cam[3] * d.x + cam[6] * d.y + cam[9] * d.z;\n"
"    float cy = cam[4] * d.x + cam[7] * d.y + cam[10] * d.z;\n"
"    float cz = cam[5] * d.x + cam[8] * d.y + cam[11] * d.z;\n"
"    ok = cz > 1e-6f;\n"
"    float halfWidth = p[32] / (2 * p[33]);\n"
"    return float2((cx / cz / cam[14] + halfWidth) * p[33] - p[34], (cy / cz / cam[14] + 0.5f) * p[33] - p[35]);\n"
"}\n"
"kernel void denoise_pack(device const float* color [[buffer(0)]], device const float* aov [[buffer(1)]],\n"
"        constant float* p [[buffer(2)]], device float* depth [[buffer(3)]],\n"
"        texture2d<float, access::write> colorT [[texture(0)]], texture2d<float, access::write> motionT [[texture(1)]],\n"
"        texture2d<float, access::write> normalT [[texture(2)]], texture2d<float, access::write> diffuseT [[texture(3)]],\n"
"        texture2d<float, access::write> specularT [[texture(4)]], texture2d<float, access::write> roughnessT [[texture(5)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    uint w = uint(p[30]);\n"
"    if (gid >= w * uint(p[31])) return;\n"
"    uint2 xy = uint2(gid % w, gid / w);\n"
"    device const float* a = aov + gid * 16;\n"
"    float z = a[3];\n"
"    float n = p[36], f = p[37], zc = max(z, n);\n"
"    depth[gid] = z <= 0 ? 1.0f : f * (zc - n) / ((f - n) * zc);\n"
"    colorT.write(float4(color[gid * 3], color[gid * 3 + 1], color[gid * 3 + 2], 1), xy);\n"
"    normalT.write(float4(a[4], a[5], a[6], 0), xy);\n"
"    diffuseT.write(float4(a[7], a[8], a[9], 1), xy);\n"
"    specularT.write(float4(a[10], a[11], a[12], 1), xy);\n"
"    roughnessT.write(float4(a[13]), xy);\n"
"    float2 motion = 0;\n"
"    if (p[38] != 0) {\n"
"        bool ok1, ok2;\n"
"        float3 pos = float3(a[0], a[1], a[2]);\n"
"        float2 now = toPixel(p, pos, z <= 0, p, ok1);\n"
"        float2 before = toPixel(p + 15, pos, z <= 0, p, ok2);\n"
"        if (ok1 && ok2) motion = before - now;\n"
"    }\n"
"    motionT.write(float4(motion, 0, 0), xy);\n"
"}\n"
"// Tone mapping like Chunky's post-processing filters: 0 gamma, 1 tonemap1, 2 ACES, 3 Hable, 4 none.\n"
"kernel void denoise_unpack(texture2d<float, access::read> output [[texture(0)]], constant float* p [[buffer(0)]],\n"
"        device uint* argb [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
"    uint w = uint(p[41]);\n"
"    if (gid >= w * uint(p[42])) return;\n"
"    float3 c = max(output.read(uint2(gid % w, gid / w)).xyz, 0.0f) * p[39];\n"
"    switch (int(p[40])) {\n"
"        case 1:\n"
"            c = max(0.0f, c - 0.004f);\n"
"            c = (c * (6.2f * c + 0.5f)) / (c * (6.2f * c + 1.7f) + 0.06f);\n"
"            break;\n"
"        case 2:\n"
"            c = clamp((c * (2.51f * c + 0.03f)) / (c * (2.43f * c + 0.59f) + 0.14f), 0.0f, 1.0f);\n"
"            c = pow(c, 1 / 2.2f);\n"
"            break;\n"
"        case 3: {\n"
"            c *= 16;\n"
"            c = ((c * (0.15f * c + 0.05f) + 0.004f) / (c * (0.15f * c + 0.5f) + 0.06f)) - 0.02f / 0.3f;\n"
"            float white = ((11.2f * (0.15f * 11.2f + 0.05f) + 0.004f) / (11.2f * (0.15f * 11.2f + 0.5f) + 0.06f)) - 0.02f / 0.3f;\n"
"            c /= white;\n"
"            break;\n"
"        }\n"
"        case 4:\n"
"            break;\n"
"        default:\n"
"            c = pow(c, 1 / 2.2f);\n"
"            break;\n"
"    }\n"
"    uint3 rgb = uint3(clamp(c * 255 + 0.5f, 0.0f, 255.0f));\n"
"    argb[gid] = 0xFF000000u | (rgb.x << 16) | (rgb.y << 8) | rgb.z;\n"
"}\n";

API_AVAILABLE(macos(26.0))
@interface CMDenoiser : NSObject
@property(nonatomic, strong) id<MTLFXTemporalDenoisedScaler> scaler;
@property(nonatomic, strong) id<MTLComputePipelineState> pack;
@property(nonatomic, strong) id<MTLComputePipelineState> unpack;
@property(nonatomic, strong) id<MTLTexture> color, depth, motion, normal, diffuse, specular, roughness, output, exposure;
@property(nonatomic, strong) id<MTLBuffer> depthBuffer;
@property(nonatomic) int inWidth, inHeight, outWidth, outHeight;
@end

@implementation CMDenoiser
@end

API_AVAILABLE(macos(26.0))
static id<MTLTexture> denoiserTexture(id<MTLDevice> device, MTLPixelFormat format, MTLTextureUsage usage, int width, int height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
            width:width height:height mipmapped:NO];
    descriptor.usage = usage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    descriptor.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:descriptor];
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_newDenoiser(JNIEnv *env, jclass cls, jlong contextHandle,
        jint inWidth, jint inHeight, jint outWidth, jint outHeight) {
    @autoreleasepool {
        if (@available(macOS 26.0, *)) {
            CMContext *context = fromHandle(contextHandle);
            id<MTLDevice> device = context.device;
            if (![MTLFXTemporalDenoisedScalerDescriptor supportsDevice:device]) return 0;
            MTLFXTemporalDenoisedScalerDescriptor *descriptor = [MTLFXTemporalDenoisedScalerDescriptor new];
            descriptor.colorTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.depthTextureFormat = MTLPixelFormatDepth32Float;
            descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
            descriptor.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.normalTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.roughnessTextureFormat = MTLPixelFormatR16Float;
            descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;
            descriptor.inputWidth = inWidth;
            descriptor.inputHeight = inHeight;
            descriptor.outputWidth = outWidth;
            descriptor.outputHeight = outHeight;
            descriptor.requiresSynchronousInitialization = YES;
            descriptor.autoExposureEnabled = NO;
            id<MTLFXTemporalDenoisedScaler> scaler = [descriptor newTemporalDenoisedScalerWithDevice:device];
            if (scaler == nil) return 0;

            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:denoiseSource options:[MTLCompileOptions new] error:&error];
            if (library == nil) {
                throwRuntime(env, [@"Metal compile failed:\n" stringByAppendingString:error.localizedDescription]);
                return 0;
            }
            CMDenoiser *denoiser = [CMDenoiser new];
            denoiser.scaler = scaler;
            denoiser.pack = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"denoise_pack"] error:&error];
            denoiser.unpack = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"denoise_unpack"] error:&error];
            if (denoiser.pack == nil || denoiser.unpack == nil) {
                throwRuntime(env, [@"Pipeline creation failed: " stringByAppendingString:error.localizedDescription]);
                return 0;
            }
            denoiser.inWidth = inWidth;
            denoiser.inHeight = inHeight;
            denoiser.outWidth = outWidth;
            denoiser.outHeight = outHeight;
            denoiser.color = denoiserTexture(device, MTLPixelFormatRGBA16Float, scaler.colorTextureUsage, inWidth, inHeight);
            denoiser.depth = denoiserTexture(device, MTLPixelFormatDepth32Float, scaler.depthTextureUsage, inWidth, inHeight);
            denoiser.motion = denoiserTexture(device, MTLPixelFormatRG16Float, scaler.motionTextureUsage, inWidth, inHeight);
            denoiser.normal = denoiserTexture(device, MTLPixelFormatRGBA16Float, scaler.normalTextureUsage, inWidth, inHeight);
            denoiser.diffuse = denoiserTexture(device, MTLPixelFormatRGBA16Float, scaler.diffuseAlbedoTextureUsage, inWidth, inHeight);
            denoiser.specular = denoiserTexture(device, MTLPixelFormatRGBA16Float, scaler.specularAlbedoTextureUsage, inWidth, inHeight);
            denoiser.roughness = denoiserTexture(device, MTLPixelFormatR16Float, scaler.roughnessTextureUsage, inWidth, inHeight);
            denoiser.output = denoiserTexture(device, MTLPixelFormatRGBA16Float, scaler.outputTextureUsage, outWidth, outHeight);
            denoiser.exposure = denoiserTexture(device, MTLPixelFormatR16Float, MTLTextureUsageShaderRead, 1, 1);
            denoiser.depthBuffer = [device newBufferWithLength:(NSUInteger) inWidth * inHeight * 4 options:MTLResourceStorageModePrivate];

            // Exposure 1: the color input is the scene's linear radiance; exposure is applied when tone mapping.
            __fp16 one = 1;
            id<MTLBuffer> exposure = [device newBufferWithBytes:&one length:sizeof(one) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            [blit copyFromBuffer:exposure sourceOffset:0 sourceBytesPerRow:2 sourceBytesPerImage:2 sourceSize:MTLSizeMake(1, 1, 1)
                       toTexture:denoiser.exposure destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            return toHandle(denoiser);
        }
        return 0;
    }
}

static void dispatch1D(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger threads) {
    NSUInteger group = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

// One frame: color and aov buffers (render kernel with CHUNKYCL_AOV, input size) to ARGB (output size).
// params: see denoiseSource; scalerParams: jitter x, y, reset history, world-to-view, view-to-clip
// (column-major). Returns the GPU time in nanoseconds.
JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_denoise(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong denoiserHandle, jlong colorHandle, jlong aovHandle, jlong argbHandle, jfloatArray params, jfloatArray scalerParams) {
    @autoreleasepool {
        if (@available(macOS 26.0, *)) {
            CMContext *context = fromHandle(contextHandle);
            CMDenoiser *denoiser = fromHandle(denoiserHandle);
            float p[43], q[35];
            (*env)->GetFloatArrayRegion(env, params, 0, 43, p);
            (*env)->GetFloatArrayRegion(env, scalerParams, 0, 35, q);
            NSUInteger inPixels = (NSUInteger) denoiser.inWidth * denoiser.inHeight;

            id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:denoiser.pack];
            [encoder setBuffer:fromHandle(colorHandle) offset:0 atIndex:0];
            [encoder setBuffer:fromHandle(aovHandle) offset:0 atIndex:1];
            [encoder setBytes:p length:sizeof(p) atIndex:2];
            [encoder setBuffer:denoiser.depthBuffer offset:0 atIndex:3];
            NSArray<id<MTLTexture>> *inputs = @[denoiser.color, denoiser.motion, denoiser.normal, denoiser.diffuse,
                                                denoiser.specular, denoiser.roughness];
            for (NSUInteger i = 0; i < inputs.count; i++) [encoder setTexture:inputs[i] atIndex:i];
            dispatch1D(encoder, denoiser.pack, inPixels);
            [encoder endEncoding];

            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            [blit copyFromBuffer:denoiser.depthBuffer sourceOffset:0 sourceBytesPerRow:denoiser.inWidth * 4
                 sourceBytesPerImage:inPixels * 4 sourceSize:MTLSizeMake(denoiser.inWidth, denoiser.inHeight, 1)
                           toTexture:denoiser.depth destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];

            id<MTLFXTemporalDenoisedScaler> scaler = denoiser.scaler;
            scaler.colorTexture = denoiser.color;
            scaler.depthTexture = denoiser.depth;
            scaler.motionTexture = denoiser.motion;
            scaler.normalTexture = denoiser.normal;
            scaler.diffuseAlbedoTexture = denoiser.diffuse;
            scaler.specularAlbedoTexture = denoiser.specular;
            scaler.roughnessTexture = denoiser.roughness;
            scaler.outputTexture = denoiser.output;
            scaler.exposureTexture = denoiser.exposure;
            scaler.jitterOffsetX = q[0];
            scaler.jitterOffsetY = q[1];
            scaler.shouldResetHistory = q[2] != 0;
            scaler.motionVectorScaleX = 1;
            scaler.motionVectorScaleY = 1;
            scaler.depthReversed = NO;
            simd_float4x4 worldToView, viewToClip;
            for (int c = 0; c < 4; c++) {
                worldToView.columns[c] = simd_make_float4(q[3 + c * 4], q[4 + c * 4], q[5 + c * 4], q[6 + c * 4]);
                viewToClip.columns[c] = simd_make_float4(q[19 + c * 4], q[20 + c * 4], q[21 + c * 4], q[22 + c * 4]);
            }
            scaler.worldToViewMatrix = worldToView;
            scaler.viewToClipMatrix = viewToClip;
            [scaler encodeToCommandBuffer:commandBuffer];

            encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:denoiser.unpack];
            [encoder setTexture:denoiser.output atIndex:0];
            [encoder setBytes:p length:sizeof(p) atIndex:0];
            [encoder setBuffer:fromHandle(argbHandle) offset:0 atIndex:1];
            dispatch1D(encoder, denoiser.unpack, (NSUInteger) denoiser.outWidth * denoiser.outHeight);
            [encoder endEncoding];
            return finish(env, commandBuffer, false);
        }
        throwRuntime(env, @"The denoiser needs macOS 26");
        return 0;
    }
}
