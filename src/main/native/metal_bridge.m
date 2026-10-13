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
        fprintf(stderr, "Metal compile warnings:\n%s\n", error.localizedDescription.UTF8String);
    }
    return library;
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compileRt(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jstring function, jboolean fastMath, jstring intersectionFunction) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return 0;
        id<MTLFunction> kernel = [library newFunctionWithName:toNSString(env, function)];
        id<MTLFunction> intersection = [library newFunctionWithName:toNSString(env, intersectionFunction)];
        if (kernel == nil || intersection == nil) {
            throwRuntime(env, @"Kernel or intersection function not found");
            return 0;
        }
        MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
        linked.functions = @[intersection];
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = kernel;
        descriptor.linkedFunctions = linked;
        NSError *error = nil;
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
}

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_compile(JNIEnv *env, jclass cls, jlong contextHandle,
        jstring source, jstring function, jboolean fastMath) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        NSString *functionName = toNSString(env, function);

        id<MTLLibrary> library = compileLibrary(env, context, source, fastMath);
        if (library == nil) return 0;
        NSError *error = nil;

        id<MTLFunction> kernel = [library newFunctionWithName:functionName];
        if (kernel == nil) {
            throwRuntime(env, [NSString stringWithFormat:@"No kernel function named %@", functionName]);
            return 0;
        }
        id<MTLComputePipelineState> pipeline = [context.device newComputePipelineStateWithFunction:kernel error:&error];
        if (pipeline == nil) {
            throwRuntime(env, [@"Pipeline creation failed: " stringByAppendingString:error.localizedDescription]);
            return 0;
        }
        return toHandle(pipeline);
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

JNIEXPORT jlong JNICALL Java_dev_thatredox_chunkynative_metal_Metal_dispatch(JNIEnv *env, jclass cls, jlong contextHandle,
        jlong pipelineHandle, jlongArray bufferHandles, jlongArray textureHandles, jint threads) {
    @autoreleasepool {
        CMContext *context = fromHandle(contextHandle);
        id pipelineObject = fromHandle(pipelineHandle);
        id<MTLComputePipelineState> pipeline = [pipelineObject isKindOfClass:[CMRtPipeline class]]
                ? ((CMRtPipeline *) pipelineObject).pipeline : pipelineObject;
        if (threads <= 0) {
            return 0;
        }

        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
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

        // Apple GPUs support non-uniform threadgroups, so the grid need not be a multiple of the group size.
        NSUInteger groupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger) threads);
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) threads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(groupSize, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            throwRuntime(env, [@"Kernel failed: " stringByAppendingString:commandBuffer.error.localizedDescription]);
            return 0;
        }
        return (jlong) ((commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1e9);
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
