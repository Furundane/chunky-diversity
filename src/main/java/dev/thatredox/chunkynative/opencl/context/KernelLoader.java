package dev.thatredox.chunkynative.opencl.context;

import org.jocl.cl_program;
import se.llbit.log.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class KernelLoader {
    private static final KernelLoader instance = new KernelLoader();
    private static final Pattern openclIncludeMatcher = Pattern.compile(
            "^\\h*#include\\h*\"\\.\\./opencl\\.h\"\\h*$", Pattern.MULTILINE);

    private final BiFunction<String, String, String> rawSourceReader;
    private final boolean hotReload;

    private KernelLoader() {
        String hotReload = System.getProperty("chunkyClHotReload");
        this.hotReload = hotReload != null;
        if (this.hotReload) {
            Path reloadPath = Paths.get(hotReload);
            rawSourceReader = (base, file) -> {
                Path filePath = reloadPath.resolve(base).resolve("include").resolve(file);
                try {
                    return new String(Files.readAllBytes(filePath));
                } catch (IOException e) {
                    Log.error(String.format("Error loading ChunkyCL, error reading file \"%s\"", filePath), e);
                    throw new IllegalStateException("Failed to load file:", e);
                }
            };
        } else {
            rawSourceReader = (base, file) -> {
                String f = base + "/" + file;
                InputStream fileStream = KernelLoader.class.getClassLoader().getResourceAsStream(f);
                if (fileStream == null) {
                    Log.errorf("Error loading ChunkyCL, file \"%s\" does not exist.", f);
                    throw new IllegalStateException(String.format("File \"%s\" does not exist.", f));
                }
                Scanner s = new Scanner(fileStream).useDelimiter("\\A");
                return s.hasNext() ? s.next() : "";
            };
        }
    }

    /**
     * Load an OpenCL program.
     *
     * @param context       OpenCL context.
     * @param base          Kernel base directory name.
     * @param kernelName    Kernel entrypoint filename.
     * @return OpenCL program.
     */
    public static cl_program loadProgram(ClContext context, String base, String kernelName) {
        return context.loadProgram(loadSource(base, kernelName));
    }

    /**
     * Load a kernel's source with all its headers inlined. The result compiles as OpenCL C, and for
     * the render kernel also as Metal Shading Language (see kernel/include/compat.h).
     *
     * @param base          Kernel base directory name.
     * @param kernelName    Kernel entrypoint filename.
     * @param defines       Preprocessor symbols to define (as 1) before the source.
     */
    public static String loadSource(String base, String kernelName, String... defines) {
        StringBuilder prefix = new StringBuilder();
        for (String define : defines) {
            prefix.append("#define ").append(define).append(" 1\n");
        }
        return prefix + inlineHeaders(file -> {
            String program = instance.rawSourceReader.apply(base, file);
            Matcher matcher = openclIncludeMatcher.matcher(program);
            program = matcher.replaceFirst("// #include \"../opencl.h\"");
            return program;
        }, kernelName, new HashSet<>());
    }

    /**
     * Replace each {@code #include "file"} line at the start of a line with the file's contents, once per file.
     * {@code #line} markers keep compiler messages pointing at the original file and line.
     */
    private static String inlineHeaders(Function<String, String> sourceReader, String file, Set<String> included) {
        included.add(file);
        String[] lines = sourceReader.apply(file).split("\\n", -1);
        StringBuilder out = new StringBuilder();
        out.append("#line 1 \"").append(file).append("\"\n");
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            // Only "quoted" includes are ours; <system> ones (Metal's metal_stdlib) stay as they are.
            if (line.startsWith("#include \"")) {
                String stripped = line.substring("#include".length()).trim();
                String header = stripped.substring(1, stripped.length() - 1);
                if (!included.contains(header)) {
                    out.append(inlineHeaders(sourceReader, header, included));
                }
                out.append("#line ").append(i + 2).append(" \"").append(file).append("\"\n");
            } else {
                out.append(line).append('\n');
            }
        }
        return out.toString();
    }

    public static boolean canHotReload() {
        return instance.hotReload;
    }
}
