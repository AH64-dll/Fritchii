# Overwolf Vulkan startup protection

On Windows, Lichii reads enabled Vulkan layer registrations in both registry
views under HKLM and HKCU before opening the Vulkan loader. Layers identified
by an Overwolf name, manifest location or library path are disabled using their
manifest's `disable_environment` controls and their exact layer names in
`VK_LOADER_LAYERS_DISABLE`. Existing loader exclusions are preserved. This runs
once per process and covers the startup probe and GUI/CLI Vulkan initialization.

No registry entries or installed files are modified. Other applications can
continue using Overwolf. A manifest's disable control can be shared by another
overlay using the same hook implementation; that control also applies within
the Lichii process. This cannot remove hooks injected before Lichii starts.

This is a process-local compatibility exclusion rather than a change to
Overwolf itself. No measured performance improvement is claimed.

The Windows translation unit was compile-checked. Focused manifest tests cover
vendor/path identification, unrelated OBS isolation, multi-layer manifests,
invalid JSON/types/environment names, empty disable values and repeat discovery.
The integrated startup implementation still needs a direct-launch check with
Overwolf installed.

Loader controls follow the [Khronos layer manifest specification](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md).
