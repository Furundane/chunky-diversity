# Measures how far preview frames move against each other (phase correlation, subpixel):
#   python3 harness/preview-shift.py <dump directory>     (frames from -Dchunkymetal.previewDump)
# Prints the shift of each frame from the previous one and from the last frame, in output pixels.
import glob
import os
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64)


def shift(a, b):
    """Translation (dx, dy) that moves image b onto image a, and the correlation peak height."""
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-12
    r = np.fft.ifft2(cross).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    # Subpixel: parabola through the peak and its neighbours on each axis.
    h, w = r.shape
    c = r[py, px]
    l, rr = r[py, (px - 1) % w], r[py, (px + 1) % w]
    u, d = r[(py - 1) % h, px], r[(py + 1) % h, px]
    sx = 0.5 * (l - rr) / (l - 2 * c + rr) if (l - 2 * c + rr) != 0 else 0.0
    sy = 0.5 * (u - d) / (u - 2 * c + d) if (u - 2 * c + d) != 0 else 0.0
    dx = (px if px <= w // 2 else px - w) + sx
    dy = (py if py <= h // 2 else py - h) + sy
    return dx, dy, c


frames = sorted(glob.glob(os.path.join(sys.argv[1], '*.png')))
images = [load(f) for f in frames]
last = images[-1]
print(f"{'frame':16} {'dx prev':>8} {'dy prev':>8} {'|d| prev':>8}   {'dx last':>8} {'dy last':>8}")
moves = []
for i, (name, img) in enumerate(zip(frames, images)):
    base = os.path.basename(name)
    lx, ly, _ = shift(last, img)
    if i == 0:
        print(f"{base:16} {'':>8} {'':>8} {'':>8}   {lx:8.3f} {ly:8.3f}")
        continue
    dx, dy, _ = shift(images[i - 1], img)
    moves.append((base, (dx * dx + dy * dy) ** 0.5))
    print(f"{base:16} {dx:8.3f} {dy:8.3f} {moves[-1][1]:8.3f}   {lx:8.3f} {ly:8.3f}")
steady = [m for b, m in moves if '-s1' in b]
if steady:
    print(f"full-resolution frames: mean move {np.mean(steady):.3f} px, max {np.max(steady):.3f} px")
