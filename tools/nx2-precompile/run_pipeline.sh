#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds a Switch 2 title's precompiled pipeline file (precompiled.fork-v<N>.nx2prep) from the
# game's own files and the Lichii disk caches recorded while playing it, and its recipe to share.
#
# usage: run_pipeline.sh CACHE.bin [CACHE.bin...]
#
# Environment:
#   NX2_GAME     the game: a .nup file (base and update), or an unpacked RomFS folder
#   NX2_OUT      output folder (default ./nx2-precompile-out)
#   NX2_RECIPE   the recipe to write (default $NX2_OUT/game.nx2recipe)
#   NX2_TOOL     nx2-precompile executable (default: nx2-precompile on PATH)
#   NX2_K        predicted render states per vertex/pixel pair (default 4)
#   NX2_CACHE_VERSION  pipeline cache version, the file name's v<N> (default: the one NX2_TOOL was
#                built with; it must match the Lichii build that reads the file)
#   PYTHON       Python 3.11+ (default python)
set -euo pipefail

: "${NX2_GAME:?set NX2_GAME to the game (.nup or an unpacked RomFS folder)}"
[ $# -ge 1 ] || { echo "usage: $0 CACHE.bin [CACHE.bin...]" >&2; exit 2; }
OUT=${NX2_OUT:-$PWD/nx2-precompile-out}
TOOL=${NX2_TOOL:-nx2-precompile}
K=${NX2_K:-4}
VERSION=${NX2_CACHE_VERSION:-$("$TOOL" cache-version)}
PREP=$OUT/precompiled.fork-v$VERSION.nx2prep
RECIPE=${NX2_RECIPE:-$OUT/game.nx2recipe}
PYTHON=${PYTHON:-python}
PY=$(cd "$(dirname "$0")" && pwd)/py
PROGRAMS=$OUT/game.programs
PRIORS=$OUT/priors.json
TRAIN=()
USED=()
for cache in "$@"; do TRAIN+=(--train "$cache"); USED+=(--used "$cache"); done
mkdir -p "$OUT"

echo "== 1. the game's shader programs, found in its files by content"
"$TOOL" extract --game "$NX2_GAME" --out "$OUT/game"

echo "== 2. storage-pointer priors from the recorded caches"
"$PYTHON" -I "$PY/learn_priors.py" "$PRIORS" "$@"

echo "== 3. oracle accuracy against each recorded cache"
for cache in "$@"; do
    "$TOOL" validate --pkg "$PROGRAMS" --cache "$cache" --out "$OUT/validate_$(basename "$cache" .bin)" \
        --priors "$PRIORS" | grep -v differs:
done

echo "== 4. every program: lowered, hashed and given oracle facts"
"$TOOL" scan --pkg "$PROGRAMS" --out "$OUT/scan" --priors "$PRIORS"

echo "== 5. inventory, vertex/pixel pairs, mesh combinations"
"$PYTHON" -I "$PY/game_tables.py" "$OUT/game.tsv" "$OUT/game.variations.tsv" "$OUT/scan.tsv" "$OUT" "${USED[@]}"
"$PYTHON" -I "$PY/mesh_draws.py" "$PROGRAMS" "$OUT/inventory.json" "$OUT/mesh_combos.json" "$OUT/mesh_draws.tsv"
"$TOOL" mesh-envs --pkg "$PROGRAMS" --cache "$OUT/mesh_draws.tsv" --out "$OUT/mesh" --priors "$PRIORS"

if [ $# -ge 2 ]; then
    echo "== 6. held-out render-state accuracy (learn from the first cache, test the second's new pipelines)"
    "$PYTHON" -I "$PY/state_eval2.py" "$OUT/inventory.json" "$OUT/nx_pairs.json" "$1" "$2"
fi

echo "== 7. precompiled file"
"$PYTHON" -I "$PY/build_precompiled.py" --inventory "$OUT/inventory.json" --envs "$OUT/scan.envs" \
    --pairs "$OUT/nx_pairs.json" "${TRAIN[@]}" --k "$K" \
    --mesh-combos "$OUT/mesh_combos.json" --mesh-envs "$OUT/mesh.mesh" --mesh-k 2 \
    --cache-version "$VERSION" --out "$PREP" --report "$OUT/precompiled_report.json"

echo "== 8. replay check"
"$TOOL" check --file "$PREP" || echo "(replay failures are prepared as failed pipelines; see above)"

echo "== 9. SPIR-V for every environment, one at a time (a compiler abort is listed, not fatal)"
"$PYTHON" -I "$PY/emit_check.py" "$TOOL" "$PREP" "$OUT/emit_crashes.json"
echo "== 10. recipe: the precompiled file without the game's code, to share; and its audit"
"$TOOL" recipe --file "$PREP" --game "$NX2_GAME" --out "$RECIPE"
"$TOOL" audit --recipe "$RECIPE" --game "$NX2_GAME"
echo "done: copy $PREP next to the title's vulkan.fork-v$VERSION.bin, or share $RECIPE"
