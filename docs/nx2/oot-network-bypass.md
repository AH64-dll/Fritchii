# Ocarina of Time: bypassing the game's anti-aliasing network

The Legend of Zelda: Ocarina of Time (`0400486024DEC000`) runs a tensor network on its tonemapped
image followed by an output pass that sharpens hard. The "Ocarina of Time anti-aliasing" setting
(`oot_anti_aliasing`: Game / No sharpening) can skip both and show the tonemapped image instead.

## What the game does

Found with sampled frame dumps (`capture_frame` with `sampled`), a trace of the texture handles in
the compute passes' constant buffer 0, and pixel statistics, in Kokiri Forest, docked, at 1x with
the built-in Max Resolution patch.

- The 3D renders at the output size (2560x1440 docked), with no jitter and no motion vectors.
- The tonemap pass (fragment shader `bded5ce917183214` outdoors, `3ac011bd2d772994` outdoors at
  another time of day, `6cb06edf14be1d79` indoors) renders a B10G11R11 image.
- The game then runs, on that **tonemapped** image: a prepass (`8475d671e6967428`), a heavy
  dispatch (shader `d9ee412f00ce399a`), the network (`42593194bb0b2b32`) and an output pass
  (`da6df3e54c716df2`) that writes the B10G11R11 image the rest of the frame uses (texture handle
  at cb0+0xd0). Keys are native program keys.
- The network keeps no history: it is a single-frame spatial smoother (mean |Laplacian| of green
  0.0042 -> 0.0037). The output pass then sharpens hard: 2.2x the input's edge contrast, with
  overshoot up to 1.72 on bright edges. That is the halo users see.
- Only the HUD and a plain copy for presentation follow the output pass.

## What No sharpening does

`ShaderCache::GetComputePipeline` asks `KeplerCompute::BypassOotNetwork` about every native compute
dispatch (`DispatchReplaced` skips it):

- the prepass and the network are skipped;
- the output pass is replaced by an `AccelerateSurfaceCopy` of the tonemapped image into the output
  pass's own target (point filter, same size; format, size and block-linear layout from the output
  pass's texture descriptor at cb0+0xd0).

The images are learned at run time, not hard-coded: the tonemapped image is render target 0 of the
latest draw whose pipeline has a tonemap fragment shader
(`RasterizerVulkan::TrackOotTonemappedImage`), and the output is whatever the output pass's handle
selects. Nothing is replaced until the tonemapped image has been seen. The prepass and the network
are skipped only while the output pass is being replaced: if the output descriptor is not the
expected B10G11R11 block-linear image or the copy fails, the next frame runs the whole network
natively again. The state lives in the rasterizer (`OotNetwork::BypassState`) because the tonemap
pass draws on the 3D engine and the network's passes do not all run on the same compute engine.

The game's constants (shader hashes, program keys, the cb0 offset) are in
`src/video_core/oot_network.h`.

## Results (Ryzen 7 9800X3D, RTX 5090, Kokiri Forest, 1x)

| Mode | fps | Image |
|---|---|---|
| Game | ~41 | sharpened, halos |
| No sharpening | ~50 | clean; edges aliased as rendered, pairs with resolution scaling |

## Caveats

- Hashes, keys and the offset are specific to the game's main with build id
  `E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348`; a game update needs them re-checked. With other
  keys the bypass does nothing.
- B10G11R11 has little precision in dark tones: without the network's smoothing some banding
  shows in deep shadows.
- The heavy dispatch between the prepass and the network still runs; its result goes unused.
- Vulkan only: other backends never learn the tonemapped image, so the setting has no effect there.
