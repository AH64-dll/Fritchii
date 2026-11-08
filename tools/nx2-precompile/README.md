# nx2-precompile

Builds Vulkan pipelines for a Switch 2 game's shaders before they are first drawn, so a new area does
not stall while the driver compiles them.

> **Fritchii.** Ported from zortos293/Lichii#895. Fritchii has no menu entry for recipes and does
> not prepare before the first frame. With `EDEN_NX2_PRECOMPILE=1` (off by default; `2` also on
> drivers without their own pipeline cache) a title's boot looks for its precompiled file, or for a
> recipe in `EDEN_NX2_PRECOMPILE_RECIPE`, the title's shader folder or `<exe dir>/recipes/`,
> assembles the recipe with the running game's files when needed and prepares the pipelines on
> idle-priority threads while the game runs (`EDEN_NX2_PRECOMPILE_THREADS`). Emulation > Prepare
> Shaders Now prepares them again. `EDEN_NX2_PRECOMPILED=0/1` below is upstream's switch and is
> not read by Fritchii. Recipes must be made against a Fritchii build: check
> `nx2-precompile cache-version` (Fritchii uses upstream's numbering since cache version 54; its
> records equal upstream's 53/54 ones), record the caches with that build
> (`vulkan-frizti.fork-v<N>.bin`), and note that Fritchii's older v52 records carry the task/mesh
> fields that upstream added in v53 (`lichii_cache.parse_cache` assumes them from v52 on).
>
> **Fritchii, zortos293/Lichii#896.** With `EDEN_NX2_PRECOMPILED_FAST=1` too (off by default) the
> preparation also keeps each pipeline's shader infos and SPIR-V in
> `precompiled_infos.fork-v<N>.bin` (written as it goes, as `.part`, and resumed with the
> checkpoint; layout in `src/video_core/renderer_vulkan/vk_precompiled_kept.h`), stamped with the
> prepared stamp, the exe's size and time, the SPIR-V settings, the graphics pipeline library split,
> the OoT workaround, `EDEN_NX2_CANONICAL_KEYS` and a hash of the other `EDEN_NX2_` switches: any
> change prepares again. A draw whose pipeline the file holds is checked against the recorded
> environments (`FileEnvironment::AnswersHoldFor`), created from the kept infos and built from the
> kept SPIR-V on a worker; anything else takes the normal path (async translate, async discard).
> Only the index stays in memory; infos, SPIR-V and recorded environments are read per pipeline.
> `EDEN_NX2_PRECOMPILED_FAST=2` also translates each such pipeline from the game's state and
> compares (the audit): one that differs is never used. #896's `EDEN_NX2_PRECOMPILED_AUDIT` and
> `EDEN_NX2_PRECOMPILED_VERIFY` are not read. `EDEN_NX2_CANONICAL_KEYS=1` (off by default) keys
> draws, the disk cache and the preparation canonically (`FixedPipelineState::CanonicalizeNative`,
> attribute types left to the keys' existing masking).

## How it works

1. `nx2-precompile extract` finds the game's shader programs in its own files, by content
   (`core/file_sys/nx2_shader_extract`): every RomFS file of the `.nup` (base and update, as the game
   sees them) is read, zstd (with the game's dictionaries) and Yaz0 are undone, SARC archives are
   expanded, and BNSH (shader archives, effect bundles, layouts), agl BAHS and NDSR packs, nn::ndve
   movie files and sead's primitive drawer are parsed. For Ocarina of Time it finds, in about 30
   seconds, every program nxnative's build of the game holds (42,848 of 42,854; the others are
   entries nxnative's current `shader_split.py` drops too) and every shader recorded sessions drew,
   and its variation tables reproduce nxnative's vertex/pixel pair tables. The cutting of programs
   and the fingerprints follow nxnative's `shader_split.py` (github.com/seedds/nxnative, MIT).
   `nx2-precompile scan` then captures and lowers each program as `GenericEnvironment::LowerNativeCode` does, and computes the shader hash
   the emulator will compute at runtime. It then translates each program against a static fact oracle
   and writes the facts the translator asked for, in `GenericEnvironment::Serialize` layout.
2. The Python tools in `py/` predict the render states each shader combination is drawn with, from
   the pipelines the recorded caches hold, and write the game's precompiled file (`NX2PREP1`).
   Vertex/pixel pairs and mesh, task and pixel combinations come from the variations the extractor
   found (`game_tables.py`; older versions of a shader archive that the game ships but does not load
   are left out), with their mesh state
   (`SET_MESH_SHADER_A/B`, `SET_MESH_INIT_SHADER` and the pass-through geometry header) derived from
   the mesh program's control section by `nx2-precompile mesh-envs`.
3. At boot, after the disk cache is loaded, `PipelineCache::PreparePrecompiled` builds every pipeline
   of that file once and drops it. The driver keeps the compiled code in its own cache. A draw still
   translates the guest program with its real facts and builds its own pipeline; when that pipeline is
   one the file predicted, the driver finds it compiled.

A native compute program's translation leaves out the accesses through a pointer slot that is null
at the time, and the program is translated again once the game fills that slot
(`PipelineCache::StaleNullPointerSlot`). Which slots are filled depends on the area: in Ocarina of
Time, the foliage culling programs. Each such state is SPIR-V the driver compiles while the game
waits, about 4 seconds for those programs. A state's translation depends only on its storage pointer
answers, so the precompiled file holds every compute program in each pointer state the recorded
caches show it in, and in the states a rebuild reaches from those (`build_precompiled.py`,
`compute_states`): each combination of the slot groups the program left null that the recordings
show filled elsewhere (slots filled together are one group), and the unions of those states. Learned
from the original three caches and one more session, the file held all six states (their SPIR-V)
that a held-out session translated those programs in, three of them rebuilds that had stalled for
about 4 seconds each, with 20 compute pipelines more.

The oracle never decides what a draw runs: a wrong guess only means the driver compiles that pipeline
when it is first drawn, as it does without this tool.

Native keys are canonical (`FixedPipelineState::CanonicalizeNative`): attribute types of inputs the
vertex program does not read, blend factors of attachments without blending, attachments past the
last colour target, and the depth bounds are cleared. None of them changes the pipeline, and the
precompiled keys can only match draws if both sides clear them. The attribute mask comes from the
vertex program's header; a program found to load an attribute its header does not list is logged,
its pipeline is dropped once, and its keys keep every attribute type from then on.

The preparation step needs a driver that keeps its own pipeline cache (NVIDIA and AMD's proprietary
drivers, as `DriverCachesPipelines` lists); elsewhere it runs only with `EDEN_NX2_PRECOMPILED=1`.
Records carry the driver of the machine that recorded them; they are prepared with this machine's,
as draws key them, so a driver update only means preparing again.

The driver's cache must be large enough to hold the whole file. **On NVIDIA, set Shader Cache Size to
Unlimited** (NVIDIA Control Panel, Manage 3D settings). With "Driver default" the driver kept about
1 GiB for the emulator: of the 132,312 pipelines prepared for Ocarina of Time it stored the first
~33,000 and silently dropped the rest, so a new area compiled almost everything again. Unlimited
held all of them (about 2.6 GB on an RTX 3080). The preparation step cannot detect this; a session's
log shows it as prepared pipelines that still take hundreds of milliseconds to build.

## Commands

```
nx2-precompile extract  --game GAME.nup|ROMFS_DIR --out PREFIX [--packages merged|union] [--threads N]
nx2-precompile scan     --pkg PREFIX.programs --out PREFIX [--priors FILE] [--threads N]
nx2-precompile validate --pkg PREFIX.programs --cache CACHE.bin --out PREFIX [--priors FILE]
nx2-precompile mesh-envs --pkg PREFIX.programs --cache DRAWS.tsv --out PREFIX [--priors FILE]
nx2-precompile check    --file precompiled.fork-v<N>.nx2prep
nx2-precompile recipe   --file precompiled.fork-v<N>.nx2prep --game GAME --out GAME.nx2recipe
nx2-precompile assemble --recipe GAME.nx2recipe --game GAME --out precompiled.fork-v<N>.nx2prep
nx2-precompile audit    --recipe GAME.nx2recipe --game GAME
nx2-precompile cache-version
```

## Recipes

A precompiled file holds the game's shader code, so it is not something to share. A recipe
(`video_core/nx2_recipe.h`) is the same file without it: for every pipeline, its render state and,
for each of its programs, a fingerprint (part of a SHA-256 of the program) and the facts its
translation asked for. A player turns it back into their own precompiled file with their own copy of
the game: `assemble`, or in Lichii the game list's "Prepare Precompiled Shaders from Recipe...". The
Ocarina of Time file of 133,016 pipelines (1.17 GB) is a recipe of 1.3 MB.

A recipe carries no program code or program header. Making one withholds, with its pipelines, any
environment whose program header is not its code's, whose mesh geometry header is neither derivable
from the code nor zero, or whose layout is unknown; and it refuses to write a recipe in which any 32
bytes of any of the game's programs appear (`Report::environments_withheld`, `FindGameCode`).
`audit` runs that last check on a recipe from elsewhere against every program of every package and
version in the game's files; run it before sharing one. What a recipe does keep from the game are
the values the translation read: constant buffer values, texture types and formats, storage buffer
locations. A recipe is made for one game version and one cache version, and is built again when
either changes. Making the same recipe twice gives the same file.

### Making a recipe for a game

One person does this per game version and cache version; everyone else only needs the recipe.

1. Build Lichii and the tool with `-DENABLE_DEBUG_TOOLS=ON` (targets `yuzu` and `nx2-precompile`).
   Check `nx2-precompile cache-version` against the Lichii build players will use; they must match.
   Install Python 3.11 or newer (no packages needed).
2. Play the game in Lichii for a while, with the NVIDIA cache size set as above. Every pipeline the
   game builds is recorded in the title's disk cache, `cache/shader/<title id>/vulkan.fork-v<N>.bin`
   in Lichii's user folder. These recorded pipelines are what the tools learn render states and
   storage-pointer facts from, so the more varied the play (areas, menus, cutscenes, effects), the
   better the predictions for the rest of the game. One long session works; caches from several
   sessions or players, in different folders, work better. Copy each cache out before the next
   session if you want to keep them apart.
3. Run the pipeline from the game's `.nup` (or an unpacked RomFS folder) and the caches, for example
   in Git Bash on Windows:

   ```
   NX2_GAME="D:/Games/Game [title][v131072].nup" NX2_OUT=out NX2_TOOL=path/to/nx2-precompile.exe \
       bash tools/nx2-precompile/run_pipeline.sh session1/vulkan.fork-v54.bin session2/vulkan.fork-v54.bin
   ```

   About 8 minutes for Ocarina of Time. It writes `out/precompiled.fork-v<N>.nx2prep` and
   `out/game.nx2recipe` (`NX2_RECIPE` names another), and ends with the audit, which must report
   0 pieces of code.
4. Optionally test it: start from an empty shader folder, prepare the recipe in Lichii (right-click
   the game, "Prepare Precompiled Shaders from Recipe..."), boot, play somewhere new, then run
   `analyze_session.sh` on that session's cache and log to see what still compiled at runtime.
5. Share the recipe, named after the game and its version (for example
   `Ocarina of Time v131072.nx2recipe`). A new game update or a Lichii build with another cache
   version needs the steps again. For a new cache version, the caches recorded under older ones
   can still be learned from (the Ocarina of Time file for v54 was built from v52 and v53 caches);
   only the precompiled file must be built by a tool of the new version.

`run_pipeline.sh CACHE.bin...` runs the whole chain from the game (`NX2_GAME`, its `.nup`) and the
disk caches recorded while playing it; see the script's header for the other variables. It needs
Python 3.11 or newer and nothing else; no keys are needed for a `.nup`. `extract` writes
PREFIX.programs (the programs, for the other commands' `--pkg`), PREFIX.tsv (fingerprint, stage and
origin of each), PREFIX.variations.tsv (the programs each variation draws together) and
PREFIX.errors.txt. `--pkg` also takes a folder of nxnative's `BTCD` packages.

`analyze_session.sh` sorts the pipelines a play session still built at runtime by cause: prepared, state not predicted, mesh, shader not in the game files.

`py/learn_priors.py` learns the storage-pointer classifications the oracle uses from the recorded
caches.

`cache-version` prints the pipeline cache version the tool was built with; the precompiled file
carries it in its name and header, and Lichii reads only a file of its own version, so a build
that changes the version needs the file built again (the log names a file of another version).

`validate` compares, for every environment a disk cache recorded, the SPIR-V from the recorded
facts with the SPIR-V from the oracle's facts, and names the fact class behind each difference.
`check` replays every environment of a precompiled file the way `FileEnvironment` does and checks
that each record's key names the code of its environments; with `NX2_CHECK_EMIT=1` it also emits
SPIR-V, and `py/emit_check.py` runs it one environment at a time so a compiler abort is listed
instead of ending the run.

`--priors` is the JSON file `py/learn_priors.py` writes: storage-pointer classifications
(`storage_pointer_direct`, `storage_pointer_indirect`) by stage, constant buffer and offset.

## Files

`PREFIX.envs` (scan): `NXPENV01`, then per program u32 ordinal, u64 shader hash, u32 size and the
serialized environment.

`PREFIX.mesh` (mesh-envs): `NXPMESH1`, then per draw u32 line, u64 mesh state hash, u64 task and
mesh shader hashes, and the sized task (possibly empty) and mesh environments.

`precompiled.fork-v<N>.nx2prep`, in the game's shader directory next to `vulkan.fork-v<N>.bin`:
`NX2PREP1`, u32 cache version, u32 environment count, u32 record count, u32 reserved; per environment
u64 offset and u64 size; per record u8 kind (0 graphics, 1 compute), u8 environment count, u16
reserved, u32 environment indices in stage order, then the 1056-byte graphics or 24-byte compute key.
Environments are stored once and shared by the records that use them.

## Switches

- `EDEN_NX2_PRECOMPILED=0`: skip the preparation step; `1`: run it also on drivers without their
  own pipeline cache.
- `EDEN_NX2_PRECOMPILED_FORCE=1`: prepare again although `precompiled_prepared.fork-v<N>.txt` says the
  file was prepared for this driver and build.
- `EDEN_NX2_EVICT_AFTER_FRAMES=<frames>`: frames a graphics pipeline may go unused before it is
  destroyed (default 9000); `0` keeps every pipeline.
- `EDEN_NX2_PIPELINE_LOG_LIMIT=<n>`: how many runtime pipeline builds over 2 ms are logged (default
  400; builds over 100 ms are always logged).
