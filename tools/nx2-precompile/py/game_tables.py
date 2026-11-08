# SPDX-License-Identifier: GPL-3.0-or-later
"""The game's shader tables, from `nx2-precompile extract` and `nx2-precompile scan`.

Writes, in OUT_DIR:
  inventory.json    one row per extracted program: its scan result (Lichii shader hash, status) and
                    where it was found (origin; fine = without the variation, coarse = the file)
  nx_pairs.json     vertex/pixel pairs drawn together: [vertex hash, pixel hash], one per pair of a
                    variation (a BNSH variation, an agl program variation, an NDSR pair)
  mesh_combos.json  [task hash or null, mesh hash, pixel hash] of every variation with a mesh shader

Unused shader archive versions: a game may ship older versions of a shader archive beside the one it
loads (Ocarina of Time: Shader/Model.Product.100 next to Model.Product.110). Archives (.bfsha) whose
paths differ only in their numbers form a group; with recorded caches (--used), the members none of
whose programs were ever drawn are left out when another member was; without them, every member but
the highest-numbered is. Other files (effects named _01, _02 ...) are never grouped.

usage: python -I game_tables.py EXTRACT.tsv EXTRACT.variations.tsv SCAN.tsv OUT_DIR [--used CACHE.bin ...]
"""
import argparse
import collections
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('extract_tsv')
ap.add_argument('variations_tsv')
ap.add_argument('scan_tsv')
ap.add_argument('out_dir')
ap.add_argument('--used', action='append', default=[], help='a recorded disk cache of the game')
args = ap.parse_args()

extracted = [line.rstrip('\n').split('\t') for line in open(args.extract_tsv, encoding='utf-8')][1:]
scanned = [line.rstrip('\n').split('\t') for line in open(args.scan_tsv, encoding='utf-8')][1:]
if len(extracted) != len(scanned):
    sys.exit(f'{args.scan_tsv} has {len(scanned)} programs, {args.extract_tsv} {len(extracted)}')
rows = []
for ordinal, (x, s) in enumerate(zip(extracted, scanned)):
    fingerprint, _, _, _, origin = x
    assert s[0] == 'programs' and int(s[1]) == ordinal, (s[:2], ordinal)
    rows.append(dict(file=s[0], index=ordinal, ordinal=ordinal, stage=s[2], status=s[3],
                     code_hash=s[4], skey=fingerprint, source=origin,
                     fine=origin.split('#var')[0], coarse=origin.split('#')[0]))

# shaders the recorded sessions drew
used = set()
for path in args.used:
    _, records = lc.parse_cache(open(path, 'rb').read(), mesh_fields=True)
    for r in records:
        if r.is_compute:
            used.add('%016x' % struct.unpack_from('<Q', r.key, 0)[0])
        else:
            used.update('%016x' % h for h in struct.unpack_from('<6Q', r.key, 0) if h)

by_fp = {r['skey']: r for r in rows}
variations = []
for line in open(args.variations_tsv, encoding='utf-8'):
    label, *fps = line.rstrip('\n').split('\t')
    variations.append((label.split('#')[0], [by_fp[f] for f in fps if f in by_fp]))

# Unused archive versions. A program both versions hold is found once, so a version counts as drawn
# by the programs only it holds.
files = collections.defaultdict(set)
for file, members in variations:
    files[file].update(m['skey'] for m in members)
groups = collections.defaultdict(set)
for file in files:
    if '.bfsha' in file:
        groups[re.sub(r'\d+', '#', file)].add(file)
left_out = set()
for name, group in groups.items():
    if len(group) < 2:
        continue
    def drawn(file):
        # by shader hash: copies that differ only outside the code lower to the same shader
        mine = {by_fp[fp]['code_hash'] for fp in files[file]}
        others = {by_fp[fp]['code_hash'] for f in group if f != file for fp in files[f]}
        return len((mine - others) & used)
    counts = {f: drawn(f) for f in group}
    if any(counts.values()):
        dropped = {f for f in group if not counts[f]}
        why = 'none of its own programs drawn, while another version\'s were'
    else:
        newest = max(group, key=lambda f: [int(n) for n in re.findall(r'\d+', f)])
        dropped = group - {newest}
        why = 'older than ' + newest
    for f in sorted(dropped):
        print(f'left out {f}: {why}')
    left_out |= dropped
kept_fps = set().union(*(files[f] for f in files if f not in left_out)) if files else set()
for r in rows:
    # programs outside every variation (none in practice) are kept
    r['left_out'] = r['skey'] not in kept_fps and any(r['skey'] in files[f] for f in left_out)

os.makedirs(args.out_dir, exist_ok=True)
json.dump(rows, open(os.path.join(args.out_dir, 'inventory.json'), 'w'))

pairs, combos = set(), set()
for file, members in variations:
    if file in left_out:
        continue
    by_stage = collections.defaultdict(list)
    for m in members:
        by_stage[m['stage']].append(m['code_hash'])
    for v in by_stage['VertexB']:
        for p in by_stage['Fragment']:
            pairs.add((v, p))
    for mesh in by_stage['Mesh']:
        for p in by_stage['Fragment']:
            for t in by_stage['Task'] or [None]:
                combos.add((t, mesh, p))
json.dump(sorted(pairs), open(os.path.join(args.out_dir, 'nx_pairs.json'), 'w'))
json.dump(sorted(combos, key=lambda c: (c[0] or '', c[1], c[2])),
          open(os.path.join(args.out_dir, 'mesh_combos.json'), 'w'))
print('programs', len(rows), 'left out', sum(r['left_out'] for r in rows), 'pairs', len(pairs),
      'mesh combinations', len(combos), 'drawn in recorded sessions', len(used))
