# Compares the GPU renders (right half) of two harness output directories:
#   python3 harness/compare-gpu.py harness/out-a harness/out-b
# The left half of each harness image is the CPU reference, which is re-rendered (with new noise)
# when it is not in the cache, so only the GPU half says whether two GPU builds render the same.
import glob
import os
import sys

from PIL import Image, ImageChops

# Cameras that are generated on the CPU with new random jitter every run.
SKIP = {'projection_FISHEYE.png', 'projection_PANORAMIC.png', 'projection_PARALLEL.png', 'dof_fisheye.png'}

a, b = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(a, '*.png'))):
    name = os.path.basename(f)
    g = os.path.join(b, name)
    if name in SKIP or not os.path.exists(g):
        continue
    A = Image.open(f).convert('RGB')
    B = Image.open(g).convert('RGB')
    w, h = A.size
    box = (w // 2 + 2, 0, w, h)
    diff = ImageChops.difference(A.crop(box), B.crop(box))
    pixels = [max(p) for p in diff.getdata()]
    rows.append((sum(1 for p in pixels if p > 2), max(pixels), name))
rows.sort(reverse=True)
print(f"{sum(1 for r in rows if r[0] == 0)} of {len(rows)} identical (GPU half, difference > 2); largest:", rows[:5])
