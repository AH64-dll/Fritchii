#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# After a play session: why did pipelines still compile at runtime?
#
# usage: analyze_session.sh CACHE.bin FIRST_NEW_RECORD eden_log.txt [NAME]
#   CACHE.bin         the title's vulkan.fork-v<N>.bin after the session
#   FIRST_NEW_RECORD  its record count before the session (printed as "records now" at the end)
#   eden_log.txt      the session's log
#
# Environment: NX2_OUT and NX2_CACHE_VERSION as for run_pipeline.sh (its outputs are read),
# NX2_TOOL, PYTHON.
set -euo pipefail

CACHE=${1:?cache file}
FIRST=${2:?first new record index}
LOG=${3:?session log}
NAME=${4:-session}
OUT=${NX2_OUT:-$PWD/nx2-precompile-out}
TOOL=${NX2_TOOL:-nx2-precompile}
PYTHON=${PYTHON:-python}
PY=$(cd "$(dirname "$0")" && pwd)/py
VERSION=${NX2_CACHE_VERSION:-$("$TOOL" cache-version)}

"$TOOL" validate --pkg "$OUT/game.programs" --cache "$CACHE" --out "$OUT/${NAME}_validate" \
    --priors "$OUT/priors.json" | grep -v differs:
"$PYTHON" -I "$PY/runtime_analysis.py" "$CACHE" "$FIRST" "$OUT/precompiled.fork-v$VERSION.nx2prep" \
    "$OUT/inventory.json" "$OUT/nx_pairs.json" "$LOG" "$OUT/${NAME}_validate.tsv" \
    "$OUT/${NAME}_analysis.json"
"$PYTHON" -I "$PY/count_records.py" "$CACHE"
