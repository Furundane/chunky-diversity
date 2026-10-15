package dev.thatredox.chunkynative.opencl.context;

import se.llbit.log.Log;
import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;

public class ClContext {
    public final Device device;
    public final cl_context context;
    public final cl_command_queue queue;
    public final cl_device_id[] deviceArray;

    public ClContext(Device device) {
        this.device = device;
        this.deviceArray = new cl_device_id[] { device.device };

        // Apple's OpenCL rejects an empty property list with CL_INVALID_VALUE.
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, device.platform);
        this.context = clCreateContext(contextProperties, 1, deviceArray,
                null, null, null);


        int[] version = device.version();
        if (version[0] >= 2) {
            cl_queue_properties queueProperties = new cl_queue_properties();
            queue = clCreateCommandQueueWithProperties(context, device.device, queueProperties, null);
        } else {
            queue = createCommandQueueOld(0);
        }

        // Check if version is behind
        if (version[0] <= 1 && version[1] < 2) {
            Log.error("OpenCL 1.2+ required.");
        }
    }

    @SuppressWarnings("deprecation")
    private cl_command_queue createCommandQueueOld(long properties) {
        return clCreateCommandQueue(context, device.device, properties, null);
    }

    /**
     * Load an OpenCL program.
     *
     * @param source  Program source with all headers inlined (see {@link KernelLoader#loadSource}).
     *                Apple's OpenCL aborts the process in clCompileProgram when headers are passed
     *                as embedded programs, so the program is built from one source string.
     * @return OpenCL program.
     */
    public cl_program loadProgram(String source) {
        cl_program kernelProgram = clCreateProgramWithSource(context, 1, new String[] { source },
                null, null);

        CL.setExceptionsEnabled(false);
        // Unsafe math optimizations (reassociation, contraction) without -cl-finite-math-only: the kernel
        // relies on inf and NaN (e.g. 1 / direction in the slab tests). About 16% faster on Apple's OpenCL
        // with the same harness results.
        String options = "-cl-std=CL1.2 -Werror -cl-unsafe-math-optimizations";
        String extensions = device.getString(CL_DEVICE_EXTENSIONS);
        if (extensions.contains("cl_khr_fp64") || extensions.contains("cl_amd_fp64")) {
            options += " -D DEVICE_FP64";
        }
        // Diagnostics: -Dchunkycl.buildOptions adds OpenCL build options.
        String extra = System.getProperty("chunkycl.buildOptions", "");
        if (!extra.isEmpty()) options += " " + extra;
        int code = clBuildProgram(kernelProgram, 1, deviceArray, options, null, null);
        if (code != CL_SUCCESS) {
            String error;
            switch (code) {
                case CL_INVALID_PROGRAM:
                    error = "CL_INVALID_PROGRAM";
                    break;
                case CL_INVALID_VALUE:
                    error = "CL_INVALID_VALUE";
                    break;
                case CL_INVALID_DEVICE:
                    error = "CL_INVALID_DEVICE";
                    break;
                case CL_INVALID_BUILD_OPTIONS:
                    error = "CL_INVALID_BUILD_OPTIONS";
                    break;
                case CL_INVALID_OPERATION:
                    error = "CL_INVALID_OPERATION";
                    break;
                case CL_COMPILER_NOT_AVAILABLE:
                    error = "CL_COMPILER_NOT_AVAILABLE";
                    break;
                case CL_BUILD_PROGRAM_FAILURE:
                    error = "CL_BUILD_PROGRAM_FAILURE";
                    break;
                case CL_OUT_OF_RESOURCES:
                    error = "CL_OUT_OF_RESOURCES";
                    break;
                case CL_OUT_OF_HOST_MEMORY:
                    error = "CL_OUT_OF_HOST_MEMORY";
                    break;
                default:
                    error = "Code " + code;
                    break;
            }
            CL.setExceptionsEnabled(true);
            Log.error("Failed to build CL program: " + error);

            long[] size = new long[1];
            clGetProgramBuildInfo(kernelProgram, deviceArray[0], CL.CL_PROGRAM_BUILD_LOG, 0, null, size);

            byte[] buffer = new byte[(int)size[0]];
            clGetProgramBuildInfo(kernelProgram, deviceArray[0], CL.CL_PROGRAM_BUILD_LOG, buffer.length, Pointer.to(buffer), null);

            throw new RuntimeException("Failed to build CL program with error: " + error + "\n" + new String(buffer, 0, buffer.length-1));
        }
        CL.setExceptionsEnabled(true);
        return kernelProgram;
    }
}
