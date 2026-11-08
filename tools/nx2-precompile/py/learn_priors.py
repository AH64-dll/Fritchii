"""Learns the oracle's storage-pointer priors from recorded disk caches.

Native programs read their buffer pointers from constant buffers, and the translator asks whether
the 8 bytes at (bank, offset) are a mapped GPU address (and, for pointer tables, those at a further
offset). The answer follows the engine's argument layout, so it repeats across programs: for each
(stage, bank, offset[, table offset]) the majority answer of the distinct recorded programs is kept
where it differs from the oracle's default (mapped).

usage: python -I learn_priors.py <out.json> <cache.bin>...
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402

STAGES = {0: 'VertexB', 1: 'TessellationControl', 2: 'TessellationEval', 3: 'Geometry',
          4: 'Fragment', 5: 'Compute', 6: 'VertexA', 7: 'Task', 8: 'Mesh'}

out_path, caches = sys.argv[1], sys.argv[2:]
votes = collections.defaultdict(collections.Counter)
seen = set()
for path in caches:
    _, recs = lc.parse_cache(open(path, 'rb').read(), mesh_fields=True)
    for rec in recs:
        for env in rec.envs:
            program = (env.stage, lc.native_shader_hash(env.code))
            for key, answer in env.storage_pointer_types:
                slot = (program, key)
                if slot in seen:
                    continue
                seen.add(slot)
                indirect = key >> 63
                bank = (key >> 32) & 0xFF
                offset = key & 0xFFFFFFFF
                table = (key >> 40) & 0x3FFFFF if indirect else None
                votes[(env.stage, indirect, bank, offset, table)][answer] += 1

priors = {'storage_pointer_direct': [], 'storage_pointer_indirect': []}
for (stage, indirect, bank, offset, table), counter in sorted(votes.items()):
    answer, _ = counter.most_common(1)[0]
    if answer == 1:
        continue
    entry = dict(stage=STAGES[stage], bank=bank, offset=offset, answer=answer,
                 votes={str(k): v for k, v in counter.items()})
    if indirect:
        entry['indirect'] = table
        priors['storage_pointer_indirect'].append(entry)
    else:
        priors['storage_pointer_direct'].append(entry)
json.dump(priors, open(out_path, 'w'), indent=0)
print('priors: %d direct, %d indirect' % (len(priors['storage_pointer_direct']),
                                          len(priors['storage_pointer_indirect'])))
