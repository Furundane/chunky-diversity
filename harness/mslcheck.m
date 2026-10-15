// Used by check-metal-kernels.py: compiles an MSL file at runtime and prints errors; with a kernel
// name, builds a pipeline linked with the listed intersection functions.
//   mslcheck <file.metal> [kernel [function ...]]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (library == nil) {
            printf("COMPILE FAILED\n%s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        if (argc > 2) {
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = [library newFunctionWithName:@(argv[2])];
            if (descriptor.computeFunction == nil) {
                printf("FAILED: no function %s\n", argv[2]);
                return 1;
            }
            NSMutableArray *functions = [NSMutableArray array];
            for (int i = 3; i < argc; i++) [functions addObject:[library newFunctionWithName:@(argv[i])]];
            if (functions.count > 0) {
                MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
                linked.functions = functions;
                descriptor.linkedFunctions = linked;
            }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:descriptor
                    options:MTLPipelineOptionNone reflection:nil error:&error];
            if (pipeline == nil) {
                printf("PIPELINE FAILED\n%s\n", error.localizedDescription.UTF8String);
                return 1;
            }
        }
        printf("OK\n");
    }
    return 0;
}
