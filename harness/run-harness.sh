#!/usr/bin/env bash
# Runs a harness task against this plugin, built against a Chunky checkout (default ../chunky next to
# this repository, or $CHUNKY_DIR). Chunky's home directory defaults to harness/chunkyhome ($CHUNKY_HOME).
#
#   ./harness/run-harness.sh runHarness                         # CPU vs GPU feature comparison (33 cases)
#   ./harness/run-harness.sh runHarness -Pcases=water_blocks,baseline_day
#   ./harness/run-harness.sh runHarness -Pgpu=ChunkyMetalRenderer -PharnessOut=out-metal
#   ./harness/run-harness.sh runGpuDebug                        # one GPU render, prints the device
#   ./harness/run-harness.sh runRealScene "-PsceneArgs=/path/to/scenes/myscene;myscene;800;800;64;512;user"
#   ./harness/run-harness.sh runProfile "-PprofileArgs=synthetic;1920;1080;300"
#   ./harness/run-harness.sh runChunky                          # Chunky GUI with the plugin
#
# Results go to harness/out. CPU reference renders are cached in harness/cache (delete it to re-render).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
exec ./gradlew -I "$REPO/harness/harness-init.gradle" -PchunkyDir="${CHUNKY_DIR:-../chunky}" \
    ${CHUNKY_HOME:+"-PchunkyHome=$CHUNKY_HOME"} --console=plain "$@"
