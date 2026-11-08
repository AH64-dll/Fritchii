# Fritchii branding

Fritchii is Frizti's fork of Lichii. Its own look (name, logo) is kept apart from emulator work so
upstream can take fixes without it.

## Telling branding from fixes

Every branding-only commit carries the trailer `Fritchii-Branding: yes` and touches nothing that
changes emulation. To list or skip them:

```sh
git log --grep='^Fritchii-Branding: yes' --oneline           # branding commits
git log --invert-grep --grep='^Fritchii-Branding: yes' ...    # everything else (fixes, ports)
```

When cherry-picking from Fritchii into Lichii, leave the branding commits out. A Claude doing the
merge can be told: "skip commits with the trailer Fritchii-Branding: yes".

## What branding changes

- `dist/fritchii.ico`, `dist/fritchii.png`: the logo (art by Frizti, Triforce badge added).
- `src/yuzu/yuzu.rc`, `src/yuzu_cmd/yuzu.rc`, `src/dedicated_room/yuzu_room.rc`, `src/yuzu/yuzu.qrc`,
  `dist/qt_themes/default/default.qrc`: point the existing icon names at the Fritchii files.
- Version string and window title "Fritchii vN" (CMakeModules/GenerateSCMRev.cmake,
  src/qt_common/qt_common.cpp, src/yuzu/bootmanager.cpp), only when dist/frizti_version.txt exists.
- This folder.

The executable stays `lichii.exe` and user data stays in the same folder, so Lichii and Fritchii
installs share settings, saves and caches.
