# chunky-diversity

Chunky's path tracer on the GPU. One kernel source runs on **OpenCL** (Windows, Linux, macOS) and **Metal**
(macOS); the goal is the same image as Chunky's CPU renderer, faster. This is a personal fork of
[ChunkyCL](https://github.com/chunky-dev/chunky-opencl), made with the help of Claude (an AI assistant).

## Supported Chunky features

"Checked" means the harness in `harness/` renders it on the CPU and the GPU and the results match, or the
setting changes the GPU image like it changes the CPU image.

| Chunky feature | GPU | Checked |
|---|---|---|
| Blocks and block models | Yes | Yes |
| Translucent blocks (glass, stained glass, panes), fancier translucency, transmissivity cap | Yes | Yes |
| Water: surfaces, refraction, opacity, visibility, custom color | Yes | Yes |
| Water shading: still, simplex, tiled normal map | Yes | Yes |
| Sky: sky light, apparent sky light, exposure, HDR skies | Yes | Yes |
| Sky modes (all are taken from Chunky's own sky) | Yes | Simulated and solid color |
| Sun: intensity, color, draw sun | Yes | Yes |
| Sun sampling: Off, Fast | Yes | Yes |
| Sun sampling: High quality | Yes | No |
| Sun sampling: Importance | Behaves like Off | |
| Emitters: on/off, intensity | Yes | Yes |
| Emitter sampling (One, All) | No | |
| Fog: uniform, layered | Yes | Yes |
| Clouds, with cloud shadows | Yes | Yes |
| Projections: pinhole, fisheye, panoramic, parallel | Yes | Yes |
| Other projections | Yes (rays from Chunky) | No |
| Depth of field | Yes | Pinhole and fisheye |
| Ray depth, target SPP, headless rendering (`chunky -render`) | Yes | Yes |
| Entities | Yes | No |
| Biome colors (grass, leaves, water) | No (fixed colors) | |
| Subsurface scattering, animated textures | No | |
| Very large worlds (NODE / BIG_PACKED octree) | No | |

The previews show the scene without fog and waves.

| Backend | Renderer | Preview |
|---|---|---|
| OpenCL | `ChunkyClRenderer` | `ChunkyClPreviewRenderer` |
| Metal (macOS) | `ChunkyMetalRenderer` | `ChunkyMetalPreviewRenderer` |

On macOS, use the Metal renderer **and** the Metal preview; mixing them uploads the scene twice.

## Tested on

| System | GPU | Backends | Tests |
|---|---|---|---|
| macOS 26.2 | Apple M4 Pro, 24 GB | Metal, Apple OpenCL | Harness (43 cases, same results on both), real worlds in the Chunky GUI |
| Windows | NVIDIA RTX 4060 | OpenCL | Harness, with an earlier version (before 0.4.0) |

Built and tested against Chunky commit [`527cb4a`](https://github.com/chunky-dev/chunky/commit/527cb4a)
(2.5.0 snapshot). On the M4 Pro, a 1024 x 1024 pass of a test scene at ray depth 20 takes about 27 ms on
Metal and 33 ms on Apple's OpenCL.

---

This is a plugin for [Chunky](https://github.com/chunky-dev/chunky) which harnesses the power of the GPU with OpenCL 1.2+ to accelerate rendering.

#### This is currently a work in progress and does not support many features. The core renderer itself is still under development so render results may change drastically between versions.

## Installation

### Note: This requires the `2.5.0` snapshots.
Download the `.jar` from the [latest release of this fork](https://github.com/Furundane/chunky-diversity/releases). In the Chunky Launcher, expand `Advanced Settings` and click on `Manage plugins`. In the `Plugin Manager` window click on `Add` and select the `.jar` file. Click on `Save` and start Chunky as usual. Remove the original ChunkyCL plugin first if you have it; the two use the same renderer names.

![image](https://user-images.githubusercontent.com/42661490/116319916-28ef2580-a76c-11eb-9f93-86d444a349fd.png)

In the `Advanced` tab, select `ChunkyClRenderer` (OpenCL) or, on macOS, `ChunkyMetalRenderer` as the renderer, and the matching preview renderer (`ChunkyClPreviewRenderer` or `ChunkyMetalPreviewRenderer`).

![image](https://user-images.githubusercontent.com/42661490/122492084-fc040580-cf99-11eb-9b08-b166dc25db41.png)

## Performance

Rough performance with a RTX 2070 is around 400 times that of the traditional CPU renderer as of 2022-01-27.

Some settings have been added to improve render performance.
* Indoor scenes should disable sunlight under `Lighting`
* Draw depth may be adjusted under `Advanced`
* Draw entities may be unchecked under `Advanced`
* OpenCL Device selector under `Advanced`

## Compatibility

* Not compatible with the Denoising Plugin.

---

## Development
`harness/` compares the GPU renderers with Chunky's CPU path tracer on small test scenes
(`./harness/run-harness.sh runHarness`, see the script for more tasks). It expects a Chunky checkout in
`../chunky`. The Metal bridge (`src/main/native`) is only built on macOS.

This project is setup to work with IntelliJ and CLion. The base directory is intended to be opened in IntelliJ and the `src/main/opencl` directory in CLion.

For hot reloading, add `-DchunkyClHotReload="<src/main/opencl directory>"` as a JVM option.

## Copyright & License
ChunkyCL is Copyright (c) 2021 - 2024, [ThatRedox](https://github.com/ThatRedox) and contributors.

Permission to modify and redistribute is granted under the terms of the GPLv3 license. See the file `LICENSE` for the full license.

ChunkyCL uses the following 3rd party libraries:
* [Chunky](https://github.com/chunky-dev/chunky/)
* [JOCL](http://www.jocl.org/) (MIT, see `licenses/jocl.txt`)
* [OpenCL header from the LLVM Project](https://llvm.org)
