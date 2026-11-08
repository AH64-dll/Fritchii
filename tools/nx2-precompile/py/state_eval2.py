"""Held-out evaluation of state_model.

usage: python -I state_eval2.py <inventory.json> <nx_pairs.json> <train.bin> <test.bin>
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402
import state_model as sm  # noqa: E402
import statefields as sf  # noqa: E402

inv_path, pairs_path, train_path, test_path = sys.argv[1:5]
inv = json.load(open(inv_path))
by_hash = collections.defaultdict(list)
for r in inv:
    by_hash[r['code_hash']].append(r)
nx_pairs = set(tuple(p) for p in json.load(open(pairs_path)))


def info(env, vertex):
    h = '%016x' % lc.native_shader_hash(env.code)
    rows = by_hash.get(h, [{}])
    d = dict(hash=h, fine=rows[0].get('fine', '?'), coarse=rows[0].get('coarse', '?'))
    if vertex:
        d['read_mask'] = sf.attribute_read_mask(env.sph)
    else:
        d['omap'] = sm.omap(env.sph)
    return d


def pipelines(path):
    _, recs = lc.parse_cache(open(path, 'rb').read(), mesh_fields=True)
    out = {}
    for r in recs:
        if r.is_compute or r.envs[0].stage != 0 or r.envs[-1].stage != 4:
            continue
        vs, fs = info(r.envs[0], True), info(r.envs[-1], False)
        state = sf.canonical(r.key[48:160], vs['read_mask'])
        out[(vs['hash'], fs['hash'], state)] = (vs, fs, state)
    return list(out.values())


train, test = pipelines(train_path), pipelines(test_path)
tk = {(v['hash'], f['hash'], s) for v, f, s in train}
test = [x for x in test if (x[0]['hash'], x[1]['hash'], x[2]) not in tk]
m = sm.Model()
for v, f, s in train:
    m.add(v, f, s)
seen_v = {v['hash'] for v, _, _ in train}
seen_f = {f['hash'] for _, f, _ in train}
KS = (1, 2, 3, 4, 6, 8)
hits, groups, pair_cov = collections.Counter(), collections.Counter(), collections.Counter()
for v, f, s in test:
    g = ('vs seen' if v['hash'] in seen_v else 'vs new') + ' / ' + ('fs seen' if f['hash'] in seen_f else 'fs new')
    groups[g] += 1
    pair_cov[(g, (v['hash'], f['hash']) in nx_pairs)] += 1
    cand = m.candidates(v, f, max(KS))
    for k in KS:
        if s in cand[:k]:
            hits[(g, k)] += 1
            hits[('ALL', k)] += 1
print('train', len(train), 'test new', len(test))
for g, n in sorted(groups.items(), key=lambda x: -x[1]):
    print('%-22s n=%5d pair-listed=%.2f ' % (g, n, pair_cov[(g, True)] / n) +
          ' '.join('@%d=%.2f' % (k, hits[(g, k)] / n) for k in KS))
n = len(test)
listed = sum(v for (g, ok), v in pair_cov.items() if ok)
print('%-22s n=%5d pair-listed=%.2f ' % ('ALL', n, listed / n) +
      ' '.join('@%d=%.3f' % (k, hits[('ALL', k)] / n) for k in KS))
