#!/usr/bin/env python3
"""Build a graphics warm-up file (vertex and fragment stages) from a static shader dump.

Graphics counterpart of make_compute_warm.py. For every unique vertex or fragment program in
the dump manifest (extract_static_shaders.py output) this lowers the native code with
shader-sweep (--capture-extent), rebuilds the code layout the emulator stores in its pipeline
cache, and writes one single-stage entry per program. Mesh and task programs are skipped.

File: 'yuzucach' + u32 version (53), then per entry (v52/v53 graphics layout):
    u32 env count = 1
    env record (GenericEnvironment::Serialize layout):
        u64 code_size, u64 texture types = 0, u64 pixel formats = 0, u64 cbuf values = N,
        u64 storage pointer types = 0, u64 cbuf replacements = 0,
        u32 local_memory_size = SPH LocalMemorySize() + common3 crs size
        u32 texture_bound     = tex cbuf index found by the lowering, else 0 (vertex) / 2 (fragment)
        u32 start_address = 0, u32 read_lowest = 0, u32 read_highest = code_size - 8
        u32 viewport_transform_state = 1, u32 stage = 0 (vertex) / 4 (fragment)
        code bytes: 80-byte SPH (first 80 bytes of the dump program), instructions with a zero
            scheduling word before every 4th word, one zero word, address map, map count, MAGIC
        N cbuf records (u64 key = 1 << 32 | offset, u32 value): constant buffer bank 1, offsets
            0..252, values = the last 256 bytes of the dump program file. Runtime records hold
            bank 1 reads exactly this way. Nothing else is recorded (no bank 0/2 values, no
            texture types, pixel formats or storage pointer types).
        80-byte SPH trailer (same bytes as the start of the code)
    1056-byte GraphicsPipelineCacheKey:
        unique_hashes[6] u64: only this stage's slot set (vertex = slot 1, fragment = slot 5),
            value = NativeShaderHash = CityHash64(code bytes) ^ 0x4f554e4345000002
        FixedPipelineState (1008 bytes) = the most common state among runtime records of the same
            stage (key bytes 56-63, color output types, zeroed: irrelevant on NVIDIA)
        vertex only: attribute types (key bytes 112-119, 2 bits per slot) = per slot, the most
            common nonzero value among runtime pipelines of the OTHER vertex shaders from the same
            game archive(s); slots with no data = 1 (float)

Contract for the emulator loader (per entry, one pipeline library per stage):
    - The stage SPIR-V inputs are the env (code, stage, local memory, texture_bound,
      viewport_transform_state, recorded bank 1 cbuf values) plus the RuntimeInfo derived from
      the key exactly as for a full pipeline (MakeRuntimeInfo with this stage only).
    - Reads that are not recorded (bank 0 and other banks, texture types, pixel formats, storage
      pointer types) must be guessed like shader-sweep --guess-missing does by default: cbuf
      bank 0 = 0x10000000, other banks = 0, texture type 3, pixel format 16, storage pointer
      true for bank 0 and false for other banks.
    - Fragment: assume the partner vertex stage writes every varying the fragment reads
      (unwritten-varying mask 0); no previous-stage info is taken from the key.
    - Entries whose stage hash the game already has in its pipeline cache can be skipped.

Validation (--validate): for dump programs whose hash equals a runtime vertex/fragment shader in
the given runtime caches, the generated entry is translated with shader-sweep --guess-missing and
its SPIR-V is compared with the runtime records' SPIR-V (real env + real key, bytes 56-63 zeroed).
The vertex attribute guess leaves the program's own runtime records out, so the rate is honest.

Needs the `cityhash` package (pip install cityhash).
"""
import argparse
import collections
import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cityhash import CityHash64

MAGIC = 0x50414d4e4544454E
HASH_XOR = 0x4F554E4345000002
ENV = struct.Struct('<6Q7I')
KEY_SIZE = 1056
STAGES = {'vertex': 0, 'fragment': 4}
SLOT = {0: 1, 4: 5}
ATTR_OFFSET = 112
BELOW_NORMAL = 0x4000 if os.name == 'nt' else 0
TIMEOUT = 300


def runtime_code(sph, instructions, offsets):
    """Same layout as LowerNativeCode (video_core/shader_environment.cpp) for graphics."""
    code = list(struct.unpack('<10Q', sph))
    for instruction in instructions:
        if len(code) % 4 == 0:
            code.append(0)  # Maxwell scheduling word
        code.append(instruction)
    code.append(0)
    if offsets:
        code.extend(0x50 + i * 8 + ((i + 1) // 3) * 8 for i in offsets)
        code.append(len(offsets))
        code.append(MAGIC)
    return struct.pack(f'<{len(code)}Q', *code)


def read_override(path):
    data = Path(path).read_bytes()
    magic, version, source_size, count, texture_buffer, map_count = struct.unpack_from('<6I', data)
    if magic != 0x4E454445 or version != 0x3230564F:
        raise ValueError(f'{path}: not an override record')
    at = 24 + source_size
    instructions = struct.unpack_from(f'<{count}Q', data, at)
    offsets = struct.unpack_from(f'<{map_count}I', data, at + count * 8)
    return instructions, offsets, texture_buffer


def local_memory(sph):
    w = struct.unpack_from('<4I', sph)
    return ((w[1] & 0xFFFFFF) | ((w[2] & 0xFFFFFF) << 24)) + (w[3] & 0xFFFFFF)


def tail_size(stage):
    return 16 if stage == 5 else 80 + (32 if stage == 3 else 0) + (92 if stage in (7, 8) else 0)


def runtime_records(path):
    """Graphics stage occurrences of a v53 cache file: (stage, hash, raw env, key)."""
    data = Path(path).read_bytes()
    cur = 12
    while cur < len(data):
        count, = struct.unpack_from('<I', data, cur)
        cur += 4
        envs = []
        for _ in range(count):
            begin = cur
            f = ENV.unpack_from(data, cur)
            p = cur + ENV.size + f[0] + f[1] * 8 + f[2] * 8 + f[3] * 12 + f[4] * 9 + f[5] * 12
            cur = p + tail_size(f[12])
            envs.append((f[12], data[begin:cur]))
        size = 24 if envs[0][0] == 5 else KEY_SIZE
        key = data[cur:cur + size]
        cur += size
        if envs[0][0] == 5:
            continue
        hashes = struct.unpack_from('<6Q', key)
        slots = [i for i in range(6) if hashes[i]]
        for slot, (stage, raw) in zip(slots, envs):
            yield stage, hashes[slot], raw, key


def normalize(key):
    k = bytearray(key)
    k[56:64] = bytes(8)  # color output types: always float on NVIDIA
    return bytes(k)


def mode(values):
    return collections.Counter(values).most_common(1)[0][0]


def make_entry(stage, code, texture_buffer, bank1, attr_types, template):
    sph = code[:80]
    bound = texture_buffer if texture_buffer != 0xFFFFFFFF else (0 if stage == 0 else 2)
    cbufs = b''.join(struct.pack('<QI', (1 << 32) | off, struct.unpack_from('<I', bank1, off)[0])
                     for off in range(0, len(bank1) & ~3, 4))
    env = ENV.pack(len(code), 0, 0, len(cbufs) // 12, 0, 0, local_memory(sph), bound, 0, 0,
                   len(code) - 8, 1, stage) + code + cbufs + sph
    key = bytearray(template[stage])
    struct.pack_into('<Q', key, SLOT[stage] * 8, CityHash64(code) ^ HASH_XOR)
    if stage == 0:
        struct.pack_into('<Q', key, ATTR_OFFSET, attr_types)
    return env, bytes(key)


def sweep(exe, scratch, tag, env, key, guess):
    """Translate one env + key; returns the SPIR-V sha1 or None, and the error text."""
    e, k, o = scratch / f'{tag}.env', scratch / f'{tag}.key', scratch / f'{tag}.spv'
    e.write_bytes(env)
    k.write_bytes(key)
    cmd = [exe, '--environment', str(e), '--cache-version', '53', '--pipeline-key', str(k),
           '--output', str(o)] + (['--guess-missing'] if guess else [])
    digest, err = None, None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                           creationflags=BELOW_NORMAL)
        result = json.loads(p.stdout)
        if result.get('status') == 'ok':
            digest = hashlib.sha1(o.read_bytes()).hexdigest()
        else:
            err = (result.get('error') or {}).get('message') or result.get('status')
    except Exception as ex:  # timeout, crash, bad output
        err = f'{type(ex).__name__}: {str(ex)[:120]}'
    for f in (e, k, o):
        if f.exists():
            f.unlink()
    return digest, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--corpus', default='./corpus',
                    help='static shader dump folder (extract_static_shaders.py output)')
    ap.add_argument('--sweep', default='shader-sweep', help='shader-sweep executable')
    ap.add_argument('--work', default='./work/graphics-ovr',
                    help='folder for the lowered override records (reused between runs)')
    ap.add_argument('--runtime-cache', nargs='+', required=True,
                    help='v53 pipeline caches recorded by the game (pipeline state statistics)')
    ap.add_argument('--out', default='./warm-graphics.bin')
    ap.add_argument('--version', type=int, default=53)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--validate', action='store_true',
                    help='compare generated entries with runtime records (translates both)')
    ap.add_argument('--scratch', default='./work/graphics-sweep')
    args = ap.parse_args()

    corpus = Path(args.corpus)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    # Runtime statistics: unique (env, key) occurrences of vertex and fragment stages.
    occurrences = {}
    for path in args.runtime_cache:
        for stage, uh, raw, key in runtime_records(path):
            if stage in SLOT:
                occurrences.setdefault((raw, key), (stage, uh))
    template = {}
    for stage in SLOT:
        key = bytearray(mode(normalize(k) for (_, k), (s, _) in occurrences.items() if s == stage))
        key[0:48] = bytes(48)
        template[stage] = bytes(key)
    runtime = collections.defaultdict(list)  # (stage, hash) -> [(raw env, key)]
    for (raw, key), (stage, uh) in occurrences.items():
        runtime[(stage, uh)].append((raw, key))

    manifest = json.loads((corpus / 'manifest.json').read_text())['programs']
    programs, skipped = {}, collections.defaultdict(set)
    for p in manifest:
        if p['stage'] in STAGES:
            programs.setdefault(p['code_sha256'], p)
        elif p['stage'] in ('mesh', 'task'):
            skipped[p['stage']].add(p['code_sha256'])

    def lower(program):
        record = work / f"{program['code_sha256']}.ovr"
        err = None
        if not record.exists():
            try:
                r = subprocess.run([args.sweep, str(corpus / program['code']), '--format',
                                    'native-header', '--capture-extent', '--lower-only',
                                    str(work / f"{program['code_sha256']}.words"),
                                    '--override-out', str(record)],
                                   capture_output=True, text=True, timeout=TIMEOUT,
                                   creationflags=BELOW_NORMAL)
                if not record.exists():
                    err = (json.loads(r.stdout).get('error') or {}).get('message')
            except Exception as ex:
                err = f'{type(ex).__name__}: {str(ex)[:120]}'
        return program, (record if record.exists() else None), err

    with ThreadPoolExecutor(args.jobs) as pool:
        lowered = list(pool.map(lower, programs.values()))

    built, failures = [], collections.Counter()
    for program, record, err in lowered:
        stage = STAGES[program['stage']]
        if record is None:
            failures[(program['stage'], err or 'no record')] += 1
            continue
        blob = (corpus / program['code']).read_bytes()
        instructions, offsets, texture_buffer = read_override(record)
        code = runtime_code(blob[:80], instructions, offsets)
        archives = {o['source'].split('romfs')[-1] for o in program['origins']}
        built.append(dict(program=program, stage=stage, code=code, tb=texture_buffer,
                          bank1=blob[-256:], uh=CityHash64(code) ^ HASH_XOR, archives=archives))

    # Vertex attribute types: runtime vertex shaders reproduced by the dump, grouped by archive.
    attr_by_uh = {}
    by_archive = collections.defaultdict(set)
    for b in built:
        if b['stage'] == 0 and (0, b['uh']) in runtime:
            attr_by_uh[b['uh']] = [struct.unpack_from('<Q', k, ATTR_OFFSET)[0]
                                   for _, k in runtime[(0, b['uh'])]]
            for a in b['archives']:
                by_archive[a].add(b['uh'])

    def attr_types(b):
        others = set().union(*(by_archive[a] for a in b['archives'])) - {b['uh']}
        counts = [collections.Counter() for _ in range(32)]
        for uh in others:
            for v in attr_by_uh[uh]:
                for i in range(32):
                    x = (v >> (2 * i)) & 3
                    if x:
                        counts[i][x] += 1
        return sum((c.most_common(1)[0][0] if c else 1) << (2 * i) for i, c in enumerate(counts))

    entries = {}
    for b in built:
        b['env'], b['key'] = make_entry(b['stage'], b['code'], b['tb'], b['bank1'],
                                        attr_types(b) if b['stage'] == 0 else 0, template)
        entries.setdefault(b['key'], b)

    out = Path(args.out)
    with open(out, 'wb') as f:
        f.write(b'yuzucach' + struct.pack('<I', args.version))
        for b in entries.values():
            f.write(struct.pack('<I', 1) + b['env'] + b['key'])
    per_stage = collections.Counter(b['program']['stage'] for b in entries.values())
    report = {'programs': {s: sum(1 for p in programs.values() if p['stage'] == s) for s in STAGES},
              'lowering_failures': {f'{s}: {e}': n for (s, e), n in failures.most_common()},
              'entries': dict(per_stage), 'skipped_unique': {s: len(v) for s, v in skipped.items()},
              'output': str(out), 'bytes': out.stat().st_size}
    print(json.dumps(report, indent=1), flush=True)

    if not args.validate:
        return 0
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    matched = {}
    for b in entries.values():
        if (b['stage'], b['uh']) in runtime:
            matched.setdefault((b['stage'], b['uh']), b)
    jobs = [(j, i, raw, normalize(key)) for j, (i, raw, key) in enumerate(
        (i, raw, key) for i, sk in enumerate(matched)
            for raw, key in runtime[sk])]
    names = list(matched)

    def reference(job):
        j, i, raw, key = job
        digest, err = sweep(args.sweep, scratch, f'r{j}', raw, key, False)
        if digest is None:  # unrecorded read in the runtime record: guess it
            digest, err = sweep(args.sweep, scratch, f'r{j}', raw, key, True)
        return i, digest

    def generated(i):
        b = matched[names[i]]
        return i, sweep(args.sweep, scratch, f'g{i}', b['env'], b['key'], True)

    with ThreadPoolExecutor(args.jobs) as pool:
        refs = collections.defaultdict(set)
        for i, digest in pool.map(reference, jobs):
            if digest:
                refs[i].add(digest)
        gens = dict(pool.map(generated, range(len(names))))
    tally = collections.defaultdict(collections.Counter)
    misses = []
    for i, (stage, uh) in enumerate(names):
        name = 'vertex' if stage == 0 else 'fragment'
        if not refs[i]:
            tally[name]['no reference'] += 1
            continue
        digest, err = gens[i]
        tally[name]['compared'] += 1
        if digest is None:
            tally[name]['generated failed'] += 1
            misses.append((name, f'{uh:016x}', err))
        elif digest in refs[i]:
            tally[name]['identical'] += 1
        else:
            misses.append((name, f'{uh:016x}', 'different SPIR-V'))
    print(json.dumps({s: dict(c) for s, c in tally.items()}, indent=1))
    print(json.dumps(collections.Counter(e for _, _, e in misses).most_common(10)))
    (scratch / 'misses.json').write_text(json.dumps(misses, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
