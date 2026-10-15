# Compiles kernel sources as Metal Shading Language and builds pipelines, to catch errors without
# running Chunky:
#   python3 harness/check-metal-kernels.py wavefront.c "CHUNKYCL_HWRT CHUNKYCL_HWRT_WATER" wf_trace wf_shadowStep
# The source is assembled like KernelLoader.loadSource (defines first, quoted includes inlined
# once). With CHUNKYCL_HWRT, pipelines are linked with the intersection function HwRt_firstBlock.
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE = os.path.join(HERE, '..', 'src', 'main', 'opencl', 'kernel', 'include')
TOOL_SOURCE = os.path.join(HERE, 'mslcheck.m')
TOOL = os.path.join(HERE, '..', 'build', 'mslcheck')


def assemble(main, defines):
    seen = set()

    def load(name):
        path = os.path.normpath(os.path.join(INCLUDE, name))
        if path in seen:
            return ''
        seen.add(path)
        src = open(path).read().replace('#include "../opencl.h"', '// #include "../opencl.h"', 1)
        out = []
        for line in src.split('\n'):
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            out.append(load(os.path.join(os.path.dirname(name), m.group(1))) if m else line)
        return '\n'.join(out)

    return ''.join('#define %s 1\n' % d for d in defines) + load(main)


def main():
    main_file, defines, kernels = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
    if not os.path.exists(TOOL) or os.path.getmtime(TOOL) < os.path.getmtime(TOOL_SOURCE):
        subprocess.check_call(['clang', '-fobjc-arc', '-framework', 'Foundation', '-framework', 'Metal',
                               TOOL_SOURCE, '-o', TOOL])
    with tempfile.NamedTemporaryFile('w', suffix='.metal', delete=False) as f:
        f.write(assemble(main_file, defines))
        source = f.name
    functions = ['HwRt_firstBlock'] if 'CHUNKYCL_HWRT' in defines else []
    failed = False
    for kernel in kernels or [None]:
        args = [TOOL, source] + ([kernel] + functions if kernel else [])
        result = subprocess.run(args, capture_output=True, text=True)
        ok = result.returncode == 0
        failed |= not ok
        print(f"{kernel or '(library)'}: {'OK' if ok else 'FAILED'}")
        if not ok:
            print('\n'.join(l for l in result.stdout.split('\n') if 'error' in l or 'FAILED' in l)[:3000])
    os.unlink(source)
    sys.exit(1 if failed else 0)


main()
