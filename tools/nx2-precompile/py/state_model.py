"""Render-state model for vertex + pixel pipelines (canonical keys, see statefields.canonical).

A state is (T, A, F, S): vertex topology/flags (T), masked attribute types (A), the render
target part (F) and a scene part (S). Each is drawn from the most specific history available:
the shader itself, its container (BNSH program set), its file, a structural class, or global.
"""
import collections
import struct

import statefields as sf


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def parts(state):
    v, f, s = sf.split(state)
    t = (v[0], v[1])
    a = int.from_bytes(v[2], 'little')
    return t, a, f, s


def assemble(t, a, f, s):
    return sf.join((t[0], t[1], a.to_bytes(8, 'little')), f, s)


def omap(sph):
    """Render targets a pixel program writes (4 bits each), from its header."""
    return u32(sph, 0x48)


class Model:
    W = {'own': 1.0, 'fine': 0.6, 'coarse': 0.25, 'class': 0.25, 'global': 0.04}

    def __init__(self):
        self.T = collections.defaultdict(collections.Counter)
        self.F = collections.defaultdict(collections.Counter)
        self.S = collections.Counter()
        self.A_own = collections.defaultdict(collections.Counter)
        self.slot = collections.defaultdict(collections.Counter)  # (container, slot) -> type

    def add(self, vs, fs, state):
        """vs/fs: dicts with hash, fine, coarse and read_mask (vertex) or omap (pixel)."""
        t, a, f, s = parts(state)
        for lvl, key in (('own', vs['hash']), ('fine', vs['fine']), ('coarse', vs['coarse']),
                         ('global', '')):
            self.T[(lvl, key)][t] += 1
        for lvl, key in (('own', fs['hash']), ('fine', fs['fine']), ('coarse', fs['coarse']),
                         ('class', fs['omap']), ('global', '')):
            self.F[(lvl, key)][f] += 1
        self.S[s] += 1
        self.A_own[vs['hash']][a] += 1
        for slot in range(32):
            if vs['read_mask'] >> slot & 1:
                ty = (a >> (2 * slot)) & 3
                for key in (vs['fine'], vs['coarse'], ''):
                    self.slot[(key, slot)][ty] += 1

    def _blend(self, table, keys, limit):
        out = collections.Counter()
        for lvl, key in keys:
            c = table.get((lvl, key))
            if not c:
                continue
            tot = sum(c.values())
            for x, n in c.most_common(limit):
                out[x] += self.W[lvl] * n / tot
            if lvl == 'own':
                break
        tot = sum(out.values()) or 1.0
        return [(x, p / tot) for x, p in out.most_common(limit)]

    def attribute_candidates(self, vs):
        own = self.A_own.get(vs['hash'])
        if own:
            tot = sum(own.values())
            return [(a, n / tot) for a, n in own.most_common(3)]
        best = 0
        floats = 0
        for slot in range(32):
            if not vs['read_mask'] >> slot & 1:
                continue
            floats |= 1 << (2 * slot)
            for key in (vs['fine'], vs['coarse'], ''):
                c = self.slot.get((key, slot))
                if c:
                    best |= c.most_common(1)[0][0] << (2 * slot)
                    break
            else:
                best |= 1 << (2 * slot)
        return [(best, 0.8)] + ([(floats, 0.2)] if floats != best else [])

    def candidates(self, vs, fs, k):
        dt = self._blend(self.T, [('own', vs['hash']), ('fine', vs['fine']),
                                  ('coarse', vs['coarse']), ('global', '')], 4)
        da = self.attribute_candidates(vs)
        df = self._blend(self.F, [('own', fs['hash']), ('fine', fs['fine']), ('coarse', fs['coarse']),
                                  ('class', fs['omap']), ('global', '')], 10)
        tot = sum(self.S.values())
        ds = [(x, n / tot) for x, n in self.S.most_common(2)]
        scored = []
        for t, pt in dt:
            for a, pa in da:
                for f, pf in df:
                    for s, ps in ds:
                        scored.append((pt * pa * pf * ps, assemble(t, a, f, s)))
        scored.sort(key=lambda x: -x[0])
        return [st for _, st in scored[:k]]
