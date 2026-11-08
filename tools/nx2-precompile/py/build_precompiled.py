"""Writes a game's precompiled pipeline file (NX2PREP1) for Lichii's PreparePrecompiled.

Records, all with canonical keys (statefields.canonical, as FixedPipelineState::CanonicalizeNative):
  * every pipeline recorded in the training caches, with its recorded environments;
  * for every vertex/pixel pair (nxnative's tables and the recorded ones), the K most likely
    render states the state model predicts, with the oracle environments of nx2-precompile;
  * every compute program, with each storage pointer state the training caches recorded it in
    (else its oracle environment) and the states a rebuild can reach from those (see
    compute_states; a compute key holds no render state).

usage: python -I build_precompiled.py --inventory inv.json --envs scan.envs --pairs nx_pairs.json
           --train a.bin [--train b.bin ...] --k 3 --cache-version N --out precompiled.fork-vN.nx2prep
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402
import state_model as sm  # noqa: E402
import statefields as sf  # noqa: E402

GRAPHICS_KEY = 1056
STATE_TAIL = GRAPHICS_KEY - 48 - sf.STATE_LEN
XFB_BIT = 1 << 6

ap = argparse.ArgumentParser()
ap.add_argument('--inventory', required=True)
ap.add_argument('--envs', required=True)
ap.add_argument('--pairs', required=True)
ap.add_argument('--train', action='append', required=True)
ap.add_argument('--k', type=int, default=3)
ap.add_argument('--out', required=True)
ap.add_argument('--report')
ap.add_argument('--mesh-combos', help='(task, mesh, pixel) combinations, mesh_variations.py')
ap.add_argument('--mesh-envs', help='NXPMESH1 file, nx2-precompile mesh-envs')
ap.add_argument('--mesh-k', type=int, default=2)
ap.add_argument('--cache-version', type=int, required=True,
                help="the Lichii build's VulkanPipelineCacheVersion (the file name's v<N>)")
args = ap.parse_args()

inv = json.load(open(args.inventory))
by_hash = collections.defaultdict(list)
for r in inv:
    by_hash[r['code_hash']].append(r)

# Oracle environments from nx2-precompile scan (one per lowered code).
oracle = {}
data = open(args.envs, 'rb').read()
assert data[:8] == b'NXPENV01'
o = 8
while o < len(data):
    ordinal, code_hash, size = struct.unpack_from('<IQI', data, o)
    o += 16
    h = '%016x' % code_hash
    oracle.setdefault(h, data[o:o + size])
    o += size
print('oracle environments', len(oracle))


def env_tail_compute(blob):
    """(workgroup, shared) from the end of a serialized compute environment."""
    wx, wy, wz, shared = struct.unpack_from('<4I', blob, len(blob) - 16)
    return (wx, wy, wz), shared


class Writer:
    def __init__(self):
        self.envs = []
        self.env_ids = {}
        self.records = []
        self.keys = set()

    def env(self, blob):
        digest = hashlib.sha256(blob).digest()
        if digest not in self.env_ids:
            self.env_ids[digest] = len(self.envs)
            self.envs.append(blob)
        return self.env_ids[digest]

    def graphics(self, blobs, key):
        assert len(key) == GRAPHICS_KEY
        h = lc.graphics_key_hash(key)
        if h in self.keys:
            return False
        self.keys.add(h)
        self.records.append((0, [self.env(b) for b in blobs], key))
        return True

    def compute(self, blob, key):
        """A compute record; one key may have several, one per storage pointer state."""
        assert len(key) == 24
        digest = (key, hashlib.sha256(blob).digest())
        if digest in self.keys:
            return False
        self.keys.add(digest)
        self.records.append((1, [self.env(blob)], key))
        return True

    def write(self, path):
        header = struct.pack('<8sIIII', b'NX2PREP1', args.cache_version, len(self.envs), len(self.records), 0)
        index_size = 16 * len(self.envs)
        body = bytearray()
        for kind, ids, key in self.records:
            body += struct.pack('<BBH', kind, len(ids), 0) + struct.pack('<%dI' % len(ids), *ids) + key
        offset = len(header) + index_size + len(body)
        index = bytearray()
        for blob in self.envs:
            index += struct.pack('<QQ', offset, len(blob))
            offset += len(blob)
        tmp = path + '.partial'
        with open(tmp, 'wb') as f:
            f.write(header)
            f.write(index)
            f.write(body)
            for blob in self.envs:
                f.write(blob)
        os.replace(tmp, path)


def unique_hashes(vs, fs, mesh_slots=None):
    hs = [0] * 6
    hs[1] = vs
    hs[5] = fs
    return struct.pack('<6Q', *hs)


def canonical_key(rec):
    """A recorded graphics key, canonical, with the bytes past the hashed state cleared."""
    key = rec.key
    vertex = next((e for e in rec.envs if e.stage == 0), None)
    mesh = any(e.stage in (7, 8) for e in rec.envs)
    mask = 0 if mesh else (sf.attribute_read_mask(vertex.sph) if vertex else 0xFFFFFFFF)
    state = sf.canonical(key[48:48 + sf.STATE_LEN], mask)
    return key[:48] + state + bytes(STATE_TAIL)


w = Writer()
model = sm.Model()
mesh_model = sm.Model()
recorded_pairs = set()
counts = collections.Counter()


def shader_info(env, vertex):
    h = '%016x' % lc.native_shader_hash(env.code)
    rows = by_hash.get(h, [{}])
    d = dict(hash=h, fine=rows[0].get('fine', '?'), coarse=rows[0].get('coarse', '?'))
    if vertex:
        d['read_mask'] = sf.attribute_read_mask(env.sph)
    else:
        d['omap'] = sm.omap(env.sph)
    return d


# A value no translation reads, which builds since cache version 54 record in each environment
# (the host properties it was made with): left out, so it does not reach the precompiled file.
READ_PROPERTIES_KEY = 0xFFFFFFFF << 32
recorded_compute = collections.defaultdict(list)   # key -> recorded environments, in cache order

for path in args.train:
    _, recs = lc.parse_cache(open(path, 'rb').read(), mesh_fields=True)
    for rec in recs:
        for e in rec.envs:
            e.cbuf_values = [(k, v) for k, v in e.cbuf_values if k != READ_PROPERTIES_KEY]
        blobs = [e.serialize(True) for e in rec.envs]
        if rec.is_compute:
            recorded_compute[rec.key].append(rec.envs[0])
            continue
        if any(e.stage in (7, 8) for e in rec.envs) and any(
                e.stage == 8 and (e.mesh_registers[0] & 0xF) != 4 for e in rec.envs):
            counts['recorded mesh skipped (topology)'] += 1
            continue
        key = canonical_key(rec)
        counts['recorded graphics'] += w.graphics(blobs, key)
        if rec.envs[0].stage == 0 and rec.envs[-1].stage == 4 and len(rec.envs) == 2:
            vs, fs = shader_info(rec.envs[0], True), shader_info(rec.envs[-1], False)
            model.add(vs, fs, key[48:48 + sf.STATE_LEN])
            recorded_pairs.add((vs['hash'], fs['hash']))
        elif any(e.stage == 8 for e in rec.envs) and rec.envs[-1].stage == 4:
            mesh_env = next(e for e in rec.envs if e.stage == 8)
            ms, fs = shader_info(mesh_env, True), shader_info(rec.envs[-1], False)
            ms['read_mask'] = 0
            mesh_model.add(ms, fs, key[48:48 + sf.STATE_LEN])

pairs = set(tuple(p) for p in json.load(open(args.pairs))) | recorded_pairs
print('pairs', len(pairs))


def oracle_info(h, vertex):
    rows = by_hash.get(h, [{}])
    blob = oracle[h]
    # The SPH sits right after the code in a serialized graphics environment; the code starts
    # with it too (lowered code begins with the header).
    sph = blob[76:76 + 0x50]
    d = dict(hash=h, fine=rows[0].get('fine', '?'), coarse=rows[0].get('coarse', '?'))
    if vertex:
        d['read_mask'] = sf.attribute_read_mask(sph)
    else:
        d['omap'] = sm.omap(sph)
    return d


for vh, fh in sorted(pairs):
    if vh not in oracle or fh not in oracle:
        counts['pair without oracle environment'] += 1
        continue
    vs, fs = oracle_info(vh, True), oracle_info(fh, False)
    added = 0
    for state in model.candidates(vs, fs, args.k + 2):
        if struct.unpack_from('<I', state, 0)[0] & XFB_BIT:
            continue
        key = unique_hashes(int(vh, 16), int(fh, 16)) + state + bytes(STATE_TAIL)
        if w.graphics([oracle[vh], oracle[fh]], key):
            added += 1
            counts['predicted graphics'] += 1
        if added >= args.k:
            break

if args.mesh_combos and args.mesh_envs:
    # Mesh draws: the mesh model knows the states mesh pipelines were drawn with; the vertex model's
    # target tables cover pixel shaders no mesh draw used yet.
    for table in ('F',):
        for key_, counter in getattr(model, table).items():
            if key_ not in getattr(mesh_model, table):
                getattr(mesh_model, table)[key_] = counter
    data = open(args.mesh_envs, 'rb').read()
    assert data[:8] == b'NXPMESH1'
    o = 8
    draws = collections.defaultdict(list)   # (task, mesh) -> [(state hash, task env, mesh env)]
    while o < len(data):
        line, state_hash, task, mesh = struct.unpack_from('<IQQQ', data, o)
        o += 28
        (tsize,) = struct.unpack_from('<I', data, o)
        tblob = data[o + 4:o + 4 + tsize]
        o += 4 + tsize
        (msize,) = struct.unpack_from('<I', data, o)
        mblob = data[o + 4:o + 4 + msize]
        o += 4 + msize
        draws[(task, mesh)].append((state_hash, tblob, mblob))
    for task_h, mesh_h, pixel_h in json.load(open(args.mesh_combos)):
        task, mesh = int(task_h, 16) if task_h else 0, int(mesh_h, 16)
        if pixel_h not in oracle or (task, mesh) not in draws:
            counts['mesh draw without environments'] += 1
            continue
        rows = by_hash.get(mesh_h, [{}])
        ms = dict(hash=mesh_h, fine=rows[0].get('fine', '?'), coarse=rows[0].get('coarse', '?'),
                  read_mask=0)
        fs = oracle_info(pixel_h, False)
        states = mesh_model.candidates(ms, fs, args.mesh_k)
        for state_hash, tblob, mblob in draws[(task, mesh)]:
            hs = [0] * 6
            hs[5] = int(pixel_h, 16)
            if task:
                hs[1], hs[3] = task ^ state_hash, mesh ^ state_hash
                blobs = [tblob, mblob, oracle[pixel_h]]
            else:
                hs[1] = mesh ^ state_hash
                blobs = [mblob, oracle[pixel_h]]
            for state in states:
                key = struct.pack('<6Q', *hs) + sf.canonical(state, 0) + bytes(STATE_TAIL)
                counts['predicted mesh'] += w.graphics(blobs, key)


# Storage pointer states of compute programs. A native compute program's translation leaves out
# the accesses through a pointer slot that was null (PipelineCache::StaleNullPointerSlot), and
# which slots the game fills depends on the area. When a slot it left out is filled, the program
# is translated again, binding every pointer the first translation bound and those filled now:
# SPIR-V the driver has not seen, compiled while the game waits (seconds for these programs). The
# translation of a state depends only on its pointer answers (the other values the recorded and the
# rebuilt environments read differ and give the same SPIR-V), so each state is a recorded
# environment with other answers to its pointer questions.
INDIRECT = 1 << 63
HOLDER_MASK = (1 << 40) - 1     # an indirect classification's key without its table offset


def null_slots(env):
    """Direct pointer slots classified as no pointer because they held 0."""
    values = dict(env.cbuf_values)
    return {k for k, v in env.storage_pointer_types
            if not k & INDIRECT and not v and values.get(k) == 0 and values.get(k + 4, 0) == 0}


# Slots seen null in one recorded compute environment and holding a pointer in another: the slots
# a game fills in some areas only. And for the pointer table entries behind a slot (an indirect
# classification), how they were answered while the slot held a pointer: some slots point to a
# table of pointers, others to a buffer.
seen_pointer, seen_null = set(), set()
table_answers = collections.defaultdict(collections.Counter)
for envs in recorded_compute.values():
    for e in envs:
        answers = dict(e.storage_pointer_types)
        seen_pointer |= {k for k, v in answers.items() if v and not k & INDIRECT}
        seen_null |= null_slots(e)
        for k, v in answers.items():
            if k & INDIRECT and answers.get(k & HOLDER_MASK):
                table_answers[k][v] += 1
fillable = seen_pointer & seen_null

# Fillable slots the game fills together (one of its lists, say): answered alike in every
# recorded environment that asks both, and both pointers in one. Each is its own group otherwise.
together = {a: {b for b in fillable if b != a} for a in fillable}
both_pointers = collections.defaultdict(bool)
for envs in recorded_compute.values():
    for e in envs:
        answers = {k: v for k, v in e.storage_pointer_types if k in fillable}
        for a, va in answers.items():
            for b in list(together[a]):
                if b in answers:
                    if answers[b] != va:
                        together[a].discard(b)
                    elif va:
                        both_pointers[(a, b)] = True
slot_group = {}
for a in sorted(fillable):
    if a not in slot_group:
        group = frozenset([a] + [b for b in together[a]
                                 if a in together[b] and both_pointers[(a, b)]])
        for b in group:
            slot_group.setdefault(b, group)
MAX_FILLED_GROUPS = 4


def filled_answer(key, value, filled):
    """A pointer answer once the slots `filled` hold pointers: a filled slot is one, a table entry
    behind it is answered as recorded while its slot held a pointer (no when never recorded)."""
    if key in filled:
        return 1
    if key & INDIRECT and key & HOLDER_MASK in filled:
        seen = table_answers.get(key)
        return seen.most_common(1)[0][0] if seen else 0
    return value
MAX_COMPUTE_STATES = 32


def compute_states(envs):
    """The environments to prepare for one compute key: the first recorded one of each set of
    storage pointer questions and answers, then the recorded one asking the most pointer questions
    (a translator of another build may ask others) with the answers of each other state a rebuild
    can reach: a recorded state with any of its groups of fillable null slots filled (slot_group,
    filled_answer), and the unions of states. Returns (environments, how many of them are recorded)."""
    base = max(reversed(envs), key=lambda e: len(e.storage_pointer_types))
    questions = [k for k, _ in base.storage_pointer_types]
    states, out, recorded_answers = [], [], set()

    def state_of(answers):
        return tuple(answers.get(k, 0) for k in questions)

    for e in envs:
        answers = tuple(sorted(e.storage_pointer_types))
        if answers not in recorded_answers:
            recorded_answers.add(answers)
            out.append(e.serialize(True))
            state = state_of(dict(answers))
            if state not in states:
                states.append(state)
    recorded_states, recorded = len(states), len(out)
    for e in list(envs):
        nulls = null_slots(e)
        # The groups all of whose slots this state left null, each filled or not.
        groups = sorted({slot_group[k] for k in nulls & fillable if slot_group[k] <= nulls},
                        key=min)
        if len(groups) > MAX_FILLED_GROUPS:
            groups = [frozenset().union(*groups)]
            counts['compute states with more null slot groups than enumerated'] += 1
        for mask in range(1, 1 << len(groups)):
            filled = frozenset().union(*(g for i, g in enumerate(groups) if mask >> i & 1))
            state = state_of({k: filled_answer(k, v, filled) for k, v in e.storage_pointer_types})
            if state not in states:
                states.append(state)
    grown = True
    while grown and len(states) < MAX_COMPUTE_STATES:
        grown = False
        for a in list(states):
            for b in list(states):
                union = tuple(max(x, y) for x, y in zip(a, b))
                if union not in states and len(states) < MAX_COMPUTE_STATES:
                    states.append(union)
                    grown = True
    if grown:
        counts['compute programs with more pointer states than prepared'] += 1
    for state in states[recorded_states:]:
        e = copy.copy(base)
        e.storage_pointer_types = list(zip(questions, state))
        out.append(e.serialize(True))
    return out, recorded


for h, blob in sorted(oracle.items()):
    rows = by_hash.get(h, [])
    if not rows or rows[0]['stage'] != 'Compute' or all(r.get('left_out') for r in rows):
        continue  # left_out: only in an archive version the game does not load (game_tables.py)
    workgroup, shared = env_tail_compute(blob)
    key = struct.pack('<QI3I', int(h, 16), shared, *workgroup)
    if key not in recorded_compute:
        recorded_compute[key] = [lc.parse_env(memoryview(blob), 0, len(blob), True)]
        counts['oracle compute'] += 1

for key, envs in recorded_compute.items():
    blobs, recorded = compute_states(envs)
    added = [w.compute(blob, key) for blob in blobs]
    counts['compute (recorded or oracle states)'] += sum(added[:recorded])
    counts['compute pointer states reached by rebuilds'] += sum(added[recorded:])
    if len(blobs) > 1:
        counts['compute programs with several pointer states'] += 1

w.write(args.out)
size = os.path.getsize(args.out)
summary = dict(counts, environments=len(w.envs), records=len(w.records), bytes=size)
print(json.dumps(summary, indent=1))
if args.report:
    json.dump(summary, open(args.report, 'w'), indent=1)
