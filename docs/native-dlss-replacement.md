# Game-DLSS replacement

The Vulkan renderer can replace a recognized game-DLSS evaluation with NVIDIA DLSS on
Windows, then AMD FSR 3.1.4 if NVIDIA DLSS is disabled or unavailable. FSR uses AMD's
eight temporal-upscaling passes, including depth, motion, history and optional RCAS.
This is separate from the presentation filters that scale the finished window image.

Fire Emblem uses the jitter-only spatial workaround by default. Explicit `fsr` and `dlss`
comparison modes enable temporal replacement for its validated input contract. The upstream
temporal blend and RCAS fallback remain available if a temporal backend is unavailable.
Invalid consumed inputs leave the native dispatch enabled. An already-rejected native
tensor shader remains an explicit dispatch failure. General tensor emulation remains.

## Supported input contract

Selection currently recognizes Fire Emblem: Fortune's Weave's captured prepass and output
shader, under its exact program ID. Temporal mode requires matching B10G11R11 input/output,
R16G16 motion, D32 depth and a fresh jitter sample. Spatial mode requires only the matching
color/output and fresh jitter. The upstream temporal contract uses reversed depth,
normalized motion toward the previous frame, HDR color and jitter in render pixels.
Resolution scaling scales the render extent and jitter together. History resets on size
changes, first use and evaluation gaps longer than 250 ms. An explicit game camera-cut
reset is not yet extracted.

Fast Fusion's shaders still execute natively. Its captured shader identities are known,
but motion scale/direction, jitter units, depth convention, reset and resource roles have
not been proven for this LLE path. The RGBA16F menu/DLAA path also requires alpha handling
that the Fire Emblem contract does not provide. Turning on NVIDIA DLSS does not override
these safety checks and is not evidence of a Fast Fusion visual fix.

For a diagnostic run, `EDEN_NX2_DLSS_CONTRACT_TRACE=1` logs at most 24 matching Fast Fusion
dispatches per compute engine. `EDEN_NX2_DLSS_CONTRACT_SKIP=N` skips the first N matching
dispatches, capped at one million. Logs include exact shader identity, QMD launch dimensions,
bounded constant-buffer words and raw texture descriptors. They contain guest addresses
and should remain private. No shaders are skipped or game files changed by this capture.

## Ocarina of Time

The Legend of Zelda: Ocarina of Time (`0400486024DEC000`) has no DLSS of its own. Its
end-of-frame upscale pass is replaced with NVIDIA DLSS when the setting is on and Streamline
has DLSS for the device. FSR is not used for this game, and no environment override is needed.

- Max Resolution add-on off: the 3/4-scale pass (`e40f5efb5480e692`) is replaced and DLSS
  writes the game's output size (1080p to 1440p docked, 810p to 1080p handheld).
- Max Resolution add-on on, docked: the full-size pass (`6bd913e1f34e83eb`) at 2560x1440 is
  replaced by a 3840x2160 DLSS image. The game still finishes its 1440p frame. At presentation,
  the HUD is taken from the difference between that frame and a 1440p copy of the DLSS scene,
  and composited over the 4K scene. The 4K image skips the presentation's post-processing and
  upscale filters.
- The game has no motion vectors or jitter. Motion comes from the scene depth reprojected
  through this and the previous frame's camera (vertex constant buffer 3). Link's recognized
  skinned draw is replayed with both poses where his depth matches. Without his pose the frame
  keeps the camera motion. The input is tonemapped, so DLSS runs in LDR without auto exposure.
- FXAA and SMAA are skipped on frames DLSS drew.
- Any unexpected input, a busy frame slot, a rescaled image (resolution scale other than 1x) or
  a failed evaluation leaves the game's pass to run. The log names the reason as
  `Ocarina DLSS: the game's upscale draws frame N (reason)`.

Shader hashes, buffer offsets and Link's mesh shape are taken from the game's current build.
A game update that changes them disables the replacement and leaves the game's pass in place.
The "Ocarina of Time anti-aliasing" option (No sharpening, docs/nx2/oot-network-bypass.md)
bypasses the same network on other GPUs. It does nothing while this replacement is enabled:
NVIDIA DLSS takes precedence because it replaces the same output pass.

## Settings and comparison runs

"Use NVIDIA DLSS in place of a game's DLSS" defaults to off and takes effect when the game
starts. Disabled means no NVIDIA runtime is loaded; an explicitly selected temporal
replacement uses FSR 3.1. Missing DLLs and unsupported GPUs fall back to FSR. Recording is
asynchronous. A failed DLSS recording clears that frame's output, restores input layouts,
and makes later evaluations fall back; it does not wait for the scheduler worker.

`EDEN_NX2_DLSS_REPLACEMENT=fsr` selects temporal replacement and forces FSR for comparison.
`dlss` selects temporal replacement with NVIDIA preferred when enabled and available.
`native` preserves the guest evaluation. With no override, Fire Emblem uses spatial de-jitter.
The old `EDEN_NX2_DLSS_STANDIN` variable is not read. The internal `standin` backend override
does not select Fire Emblem's intercepted dispatch. The RCAS sharpness setting controls FSR
and fallback sharpening. NVIDIA model-selection overrides
from the upstream implementation remain available for diagnostic comparisons.

## Build and distribution

FSR needs no external runtime. Keep `externals/FidelityFX-SDK/LICENSE.txt` and its attribution
with distributions. The vendored files are unmodified AMD MIT sources from FidelityFX SDK
v1.1.4, commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`.

For the optional Windows NVIDIA path, obtain only the MIT Streamline headers and license
from NVIDIA-RTX/Streamline tag `v2.14.16`, commit
`625db07d4bca5abc67d781c42af7e7fb0761f1b6`. The SDK directory must contain `include/sl.h`,
`include/sl_dlss.h`, `include/sl_helpers_vk.h`, `include/sl_security.h` and their companion
headers. MSVC x86_64 builds enable `ENABLE_NVIDIA_DLSS` by default and fetch the headers;
`-DEGLANTINE_STREAMLINE_SDK=<header-root>` selects a local copy. This adds Windows
`version`, `wintrust` and `crypt32` libraries, not NVIDIA import libraries. Retain Streamline's
MIT `license.txt` in the packaged notices.

Do not bundle NVIDIA's proprietary DLLs. The Graphics panel obtains missing files from
NVIDIA's Streamline release only after the user reviews and accepts the license texts.
The runtime searches `EDEN_STREAMLINE_DIR`, the user `streamline` directory, then the
executable's `streamline` directory. It verifies the interposer signature and supports the
NVIDIA app's signed DLSS model. The UI and Windows loading path require Windows validation.

## Provenance

This backport is AI-assisted and requires human review and testing before distribution.
It carries upstream work by Duck912 from Eglantine commits `08dbcc7e3b3d`, `11cebf56014a`
and PR #434 through `f9d0c802e8e4`, plus Jonas's warning fix `537c1f4e173a`. The empty-extension requirement
fix `1889602c7a0d` is a separately included follow-up. Existing SPDX notices and AMD/NVIDIA
license text are retained. The upstream nx2 merge also carries newer asynchronous recording
and fallback changes while retaining the fork's signature checks and input validation.
