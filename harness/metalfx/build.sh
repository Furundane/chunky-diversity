#!/usr/bin/env bash
# Builds the MetalFX denoiser experiment library (arm64, macOS 26 SDK) into build/libmetalfxdenoiser.dylib.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../build/metalfx"
JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home)}"
mkdir -p "$OUT"
clang -fobjc-arc -dynamiclib -O2 -Wall -Werror -arch arm64 -mmacosx-version-min=26.0 \
    -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/darwin" \
    -framework Foundation -framework Metal -framework MetalFX \
    "$DIR/metalfx_denoiser.m" -o "$OUT/libmetalfxdenoiser.dylib"
echo "$OUT/libmetalfxdenoiser.dylib"
