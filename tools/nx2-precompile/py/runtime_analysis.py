"""Why did pipelines still compile at runtime? Compares the records a session appended to the disk
cache (real facts, canonical keys) with the precompiled file, and joins the log's build times.

usage: python -I runtime_analysis.py <cache.bin> <first new record> <precompiled.nx2prep>
           <inventory.json> <nx_pairs.json> <eden_log.txt> <validate.tsv> <out.json>
"""
import collections
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402
import state_model as sm  # noqa: E402
import statefields as sf  # noqa: E402

cache_path, first_new, prep_path, inv_path, pairs_path, log_path, validate_path, out_path = sys.argv[1:9]
first_new = int(first_new)

# ---- precompiled file: keys and, per shader pair, the states it holds
data = open(prep_path, 'rb').read()
magic, version, env_count, record_count, _ = struct.unpack_from('<8sIIII', data, 0)
off = 24 + 16 * env_count
prep_graphics, prep_compute = set(), set()
prep_states = collections.defaultdict(set)
for _ in range(record_count):
    kind, count, _ = struct.unpack_from('<BBH', data, off)
    off += 4 + 4 * count
    if kind == 1:
        prep_compute.add(data[off:off + 24])
        off += 24
    else:
        key = data[off:off + 1056]
        prep_graphics.add(lc.graphics_key_hash(key))
        hs = struct.unpack_from('<6Q', key, 0)
        prep_states[(hs[1], hs[5])].add(key[48:160])
        off += 1056
del data

inv = json.load(open(inv_path))
by_hash = {}
for r in inv:
    by_hash.setdefault(r['code_hash'], r)
pairs = set(tuple(p) for p in json.load(open(pairs_path)))

# ---- oracle SPIR-V agreement for the session's environments (nx2-precompile validate)
spirv = {}
for line in open(validate_path).read().splitlines()[1:]:
    st, h, res, det = (line.split('\t') + [''])[:4]
    spirv.setdefault((st, h), (res, det))

# ---- build times from the log (the first 400 per session are logged individually)
times = {}
pat = re.compile(r'(Graphics|Compute) pipeline built on the GPU thread in ([\d.]+) ms: pipeline=([0-9a-f]+)')
summary = re.compile(r'Runtime pipelines in the last 10 s: (\d+) built, ([\d.]+) ms on the GPU thread.*?([\d.]+) ms of draws held')
windows = []
for line in open(log_path, encoding='utf-8', errors='replace'):
    m = pat.search(line)
    if m:
        times[int(m.group(3), 16)] = float(m.group(2))
    m = summary.search(line)
    if m:
        t = float(line[1:line.index(']')])
        windows.append((t, int(m.group(1)), float(m.group(2)), float(m.group(3))))

STAGE = {0: 'VertexB', 4: 'Fragment', 5: 'Compute', 7: 'Task', 8: 'Mesh'}


def part_diff(state, candidates):
    """Which state parts differ from the closest precompiled state of the same pair."""
    v, f, s = sf.split(state)
    t, a = (v[0], v[1]), v[2]
    best = None
    for c in candidates:
        cv, cf, cs = sf.split(c)
        diff = []
        if (cv[0], cv[1]) != t:
            diff.append('topology/flags')
        if cv[2] != a:
            diff.append('attribute types')
        if cf != f:
            fdiff = []
            if cf[2] != f[2]:
                fdiff.append('colour formats')
            if cf[4] != f[4]:
                fdiff.append('blending')
            if cf[1] != f[1] or cf[0] != f[0]:
                fdiff.append('depth/msaa/alpha')
            diff.append('target(' + '+'.join(fdiff or ['other']) + ')')
        if cs != s:
            diff.append('scene')
        if best is None or len(diff) < len(best):
            best = diff
    return ', '.join(best) if best else 'no candidates'


_, recs = lc.parse_cache(open(cache_path, 'rb').read(), mesh_fields=True)
rows = []
for rec in recs:
    if rec.index < first_new:
        continue
    hashes = ['%016x' % lc.native_shader_hash(e.code) for e in rec.envs]
    stages = [STAGE.get(e.stage, str(e.stage)) for e in rec.envs]
    agree = [spirv.get((st, h), ('?', ''))[0] for st, h in zip(stages, hashes)]
    known = [h in by_hash for h in hashes]
    if rec.is_compute:
        key_hash = None
        prepared = rec.key in prep_compute
        why = 'compute: prepared, facts differ' if prepared and 'spirv-differs' in agree else \
              'compute: prepared (driver hit expected)' if prepared else \
              'compute: shader not in game files' if not all(known) else 'compute: not prepared'
        detail = ''
    else:
        key_hash = lc.graphics_key_hash(rec.key)
        mesh = any(e.stage in (7, 8) for e in rec.envs)
        prepared = key_hash in prep_graphics
        if prepared:
            why = 'prepared, but facts differ (SPIR-V)' if 'spirv-differs' in agree else 'prepared (driver hit expected)'
            detail = ''
        elif mesh:
            why, detail = 'mesh pipeline (not predicted)', ''
        elif not all(known):
            why, detail = 'shader not in game files', ''
        else:
            vh, fh = int(hashes[0], 16), int(hashes[-1], 16)
            if (hashes[0], hashes[-1]) not in pairs and (vh, fh) not in prep_states:
                why, detail = 'shader pair not listed', ''
            else:
                why = 'state not among predictions'
                detail = part_diff(rec.key[48:160], prep_states.get((vh, fh), ()))
    ms = times.get(key_hash) if key_hash is not None else None
    rows.append(dict(index=rec.index, why=why, detail=detail, stages=stages, hashes=hashes,
                     spirv=agree, ms=ms, key=('%016x' % key_hash) if key_hash else None))

by_why = collections.Counter(r['why'] for r in rows)
by_detail = collections.Counter(r['detail'] for r in rows if r['detail'])
timed = collections.defaultdict(list)
for r in rows:
    if r['ms'] is not None:
        timed[r['why']].append(r['ms'])
print('new runtime records', len(rows))
for why, n in by_why.most_common():
    t = sorted(timed.get(why, []))
    med = '%.1f ms median of %d logged, %d over 30 ms' % (t[len(t) // 2], len(t), sum(1 for x in t if x > 30)) if t else 'no logged times'
    print('  %5d  %-45s %s' % (n, why, med))
print('state misses by differing part:')
for d, n in by_detail.most_common(12):
    print('  %5d  %s' % (n, d))
print('10 s windows: %d, builds %d, GPU-thread %.0f ms, draws held %.0f ms' % (
    len(windows), sum(w[1] for w in windows), sum(w[2] for w in windows), sum(w[3] for w in windows)))
for w in sorted(windows, key=lambda w: -w[3])[:5]:
    print('  t=%.0f s: %d built, %.0f ms GPU thread, %.0f ms held' % w)
json.dump(dict(rows=rows, by_why=by_why, by_detail=by_detail, windows=windows), open(out_path, 'w'), indent=1)
