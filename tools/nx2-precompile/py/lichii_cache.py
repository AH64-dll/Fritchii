#!/usr/bin/env python3
"""Reader/writer for Lichii (Eden/yuzu fork, nx2 branch) Vulkan shader-environment disk caches.

Layout derived from Lichii commit 588bfa611b (VulkanPipelineCacheVersion = 53):
  src/video_core/shader_environment.cpp  GenericEnvironment::Serialize (292-353),
                                         FileEnvironment::Deserialize (1048-1126),
                                         SerializePipeline (1235-1265), LoadPipelines (1267-1322)
  src/video_core/renderer_vulkan/vk_pipeline_cache.cpp  load_compute / load_graphics (~890-998),
                                         LoadUsedPipelines / SaveUsedPipelines (669-775)
  src/video_core/renderer_vulkan/fixed_pipeline_state.h FixedPipelineState, Size()
  src/video_core/renderer_vulkan/vk_graphics_pipeline.h GraphicsPipelineCacheKey (Size(), Hash)

All integers little-endian. No checksums, no per-record length prefix.

File:   char magic[8] = "yuzucach"; u32 version; then records until EOF.
Record: u32 num_envs; Env[num_envs]; key bytes
        key = ComputePipelineCacheKey (24 B) if envs[0].stage == Compute(5)
              else GraphicsPipelineCacheKey (1056 B = 48 B unique_hashes + 1008 B FixedPipelineState)
Env:    u64 code_size, n_texture_types, n_texture_pixel_formats, n_cbuf_values,
            n_storage_pointer_types, n_cbuf_replacements
        u32 local_memory_size, texture_bound, start_address, cached_lowest (-> read_lowest),
            cached_highest (-> read_highest), viewport_transform_state, stage
        u8  code[code_size]
        n_texture_types         x (u32 handle, u32 Shader::TextureType)
        n_texture_pixel_formats x (u32 handle, i32 Shader::TexturePixelFormat)
        n_cbuf_values           x (u64 (index<<32|offset), u32 value)
        n_storage_pointer_types x (u64 key, u8 mapped)
        n_cbuf_replacements     x (u64 (bank<<32|offset), u32 Shader::ReplaceConstant)
        if stage == Compute: u32 workgroup_size[3]; u32 shared_memory_size
        else: ProgramHeader sph (80 B)
              if stage == Geometry(3): u32 gp_passthrough_mask[8]
              if stage in (Task(7), Mesh(8)): u32 mesh_registers[3]; ProgramHeader mesh_primitive_header (80 B)
                                              (added in v53, commit b06d1818ee; absent in v52)
vulkan_used: u32 magic 0x31505545 ("EUP1"); u32 count; count x (u64 GraphicsPipelineCacheKey::Hash(), u8 age)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator

MASK = (1 << 64) - 1
MAGIC = b"yuzucach"
USED_MAGIC = 0x31505545
NATIVE_MAP_MAGIC = 0x50414D4E4544454E
NATIVE_HASH_TAG = 0x4F554E4345000002
SPH_SIZE = 0x50
COMPUTE_KEY_SIZE = 24
GRAPHICS_KEY_SIZE = 1056
FPS_SIZE = 1008  # sizeof(FixedPipelineState)
# offsets inside FixedPipelineState (computed from fixed_pipeline_state.h member order/sizes)
FPS_OFF = {
    "raw1": 0, "raw2": 4, "color_formats": 8, "driver_id": 16, "driver_version": 20,
    "alpha_test_ref": 24, "point_size": 28, "viewport_swizzles": 32, "attribute_types": 64,
    "dynamic_state": 72, "attachments": 80, "attributes": 112, "binding_divisors": 240,
    "vertex_strides": 368, "xfb_state": 432, "depth_bounds_min": 992, "depth_bounds_max": 996,
    "line_stipple_factor": 1000, "line_stipple_pattern": 1004,
}
STAGES = ("VertexB", "TessellationControl", "TessellationEval", "Geometry", "Fragment",
          "Compute", "VertexA", "Task", "Mesh")
STAGE_COMPUTE, STAGE_GEOMETRY, STAGE_TASK, STAGE_MESH = 5, 3, 7, 8
# Maxwell::ShaderType slot -> Shader::Stage values allowed for an env in that slot
SLOT_STAGES = {0: {6}, 1: {0, 7, 8}, 2: {1}, 3: {2, 8}, 4: {3}, 5: {4}}

ENV_HDR = struct.Struct("<6Q7I")  # 76 bytes


# ---------------------------------------------------------------- CityHash64 (v1.1, as in src/common/cityhash.cpp)
_k0 = 0xC3A5C85C97CB3127
_k1 = 0xB492B66FBE98F273
_k2 = 0x9AE16A3B2F90404F


def _rot(v, s):
    return v if s == 0 else ((v >> s) | (v << (64 - s))) & MASK


def _smix(v):
    return v ^ (v >> 47)


def _h16(u, v, mul=0x9DDFEA08EB382D69):
    a = ((u ^ v) * mul) & MASK
    a ^= a >> 47
    b = ((v ^ a) * mul) & MASK
    b ^= b >> 47
    return (b * mul) & MASK


def _bswap(v):
    return int.from_bytes(v.to_bytes(8, "little"), "big")


def cityhash64(s: bytes) -> int:
    n = len(s)
    f64 = lambda o: int.from_bytes(s[o:o + 8], "little")
    f32 = lambda o: int.from_bytes(s[o:o + 4], "little")
    if n <= 16:
        if n >= 8:
            mul = (_k2 + n * 2) & MASK
            a = (f64(0) + _k2) & MASK
            b = f64(n - 8)
            c = (_rot(b, 37) * mul + a) & MASK
            d = ((_rot(a, 25) + b) * mul) & MASK
            return _h16(c, d, mul)
        if n >= 4:
            mul = (_k2 + n * 2) & MASK
            a = f32(0)
            return _h16((n + (a << 3)) & MASK, f32(n - 4), mul)
        if n > 0:
            a, b, c = s[0], s[n >> 1], s[n - 1]
            y = a + (b << 8)
            z = n + (c << 2)
            return (_smix(((y * _k2) ^ (z * _k0)) & MASK) * _k2) & MASK
        return _k2
    if n <= 32:
        mul = (_k2 + n * 2) & MASK
        a = (f64(0) * _k1) & MASK
        b = f64(8)
        c = (f64(n - 8) * mul) & MASK
        d = (f64(n - 16) * _k2) & MASK
        return _h16((_rot((a + b) & MASK, 43) + _rot(c, 30) + d) & MASK,
                    (a + _rot((b + _k2) & MASK, 18) + c) & MASK, mul)
    if n <= 64:
        mul = (_k2 + n * 2) & MASK
        a = (f64(0) * _k2) & MASK
        b = f64(8)
        c = f64(n - 24)
        d = f64(n - 32)
        e = (f64(16) * _k2) & MASK
        f = (f64(24) * 9) & MASK
        g = f64(n - 8)
        h = (f64(n - 16) * mul) & MASK
        u = (_rot((a + g) & MASK, 43) + (_rot(b, 30) + c) * 9) & MASK
        v = (((a + g) & MASK) ^ d) + f + 1 & MASK
        w = (_bswap(((u + v) * mul) & MASK) + h) & MASK
        x = (_rot((e + f) & MASK, 42) + c) & MASK
        y = ((_bswap(((v + w) * mul) & MASK) + g) * mul) & MASK
        z = (e + f + c) & MASK
        a = (_bswap(((x + z) * mul + y) & MASK) + b) & MASK
        b = (_smix(((z + a) * mul + d + h) & MASK) * mul) & MASK
        return (b + x) & MASK
    words = struct.unpack_from("<%dQ" % (n // 8), s) if n % 8 == 0 else None

    def W(o):  # fetch64 at byte offset o
        if words is not None and o % 8 == 0:
            return words[o >> 3]
        return int.from_bytes(s[o:o + 8], "little")

    def weak(o, a, b):
        w_, x_, y_, z_ = W(o), W(o + 8), W(o + 16), W(o + 24)
        a = (a + w_) & MASK
        b = _rot((b + a + z_) & MASK, 21)
        c = a
        a = (a + x_ + y_) & MASK
        b = (b + _rot(a, 44)) & MASK
        return (a + z_) & MASK, (b + c) & MASK

    x = W(n - 40)
    y = (W(n - 16) + W(n - 56)) & MASK
    z = _h16((W(n - 48) + n) & MASK, W(n - 24))
    v = weak(n - 64, n, z)
    w = weak(n - 32, (y + _k1) & MASK, x)
    x = (x * _k1 + W(0)) & MASK
    ln = (n - 1) & ~63
    o = 0
    while True:
        x = (_rot((x + y + v[0] + W(o + 8)) & MASK, 37) * _k1) & MASK
        y = (_rot((y + v[1] + W(o + 48)) & MASK, 42) * _k1) & MASK
        x ^= w[1]
        y = (y + v[0] + W(o + 40)) & MASK
        z = (_rot((z + w[0]) & MASK, 33) * _k1) & MASK
        v = weak(o, (v[1] * _k1) & MASK, (x + w[0]) & MASK)
        w = weak(o + 32, (z + w[1]) & MASK, (y + W(o + 16)) & MASK)
        z, x = x, z
        o += 64
        ln -= 64
        if ln == 0:
            break
    return _h16((_h16(v[0], w[0]) + _smix(y) * _k1 + z) & MASK, (_h16(v[1], w[1]) + x) & MASK)


def native_shader_hash(code: bytes) -> int:
    return cityhash64(code) ^ NATIVE_HASH_TAG


# ---------------------------------------------------------------- data model
@dataclass
class Env:
    offset: int
    end: int
    code_size: int
    local_memory_size: int
    texture_bound: int
    start_address: int
    cached_lowest: int
    cached_highest: int
    viewport_transform_state: int
    stage: int
    code: bytes
    texture_types: list
    texture_pixel_formats: list
    cbuf_values: list
    storage_pointer_types: list
    cbuf_replacements: list
    workgroup_size: tuple | None = None
    shared_memory_size: int | None = None
    sph: bytes | None = None
    gp_passthrough_mask: tuple | None = None
    mesh_registers: tuple | None = None
    mesh_primitive_header: bytes | None = None

    @property
    def has_native_map(self) -> bool:
        return len(self.code) >= 16 and int.from_bytes(self.code[-8:], "little") == NATIVE_MAP_MAGIC

    def native_map_count(self) -> int | None:
        if not self.has_native_map:
            return None
        return int.from_bytes(self.code[-16:-8], "little")

    def serialize(self, with_mesh_fields: bool = True) -> bytes:
        out = bytearray(ENV_HDR.pack(
            self.code_size, len(self.texture_types), len(self.texture_pixel_formats),
            len(self.cbuf_values), len(self.storage_pointer_types), len(self.cbuf_replacements),
            self.local_memory_size, self.texture_bound, self.start_address, self.cached_lowest,
            self.cached_highest, self.viewport_transform_state, self.stage))
        out += self.code
        for k, t in self.texture_types:
            out += struct.pack("<II", k, t)
        for k, t in self.texture_pixel_formats:
            out += struct.pack("<Ii", k, t)
        for k, t in self.cbuf_values:
            out += struct.pack("<QI", k, t)
        for k, t in self.storage_pointer_types:
            out += struct.pack("<QB", k, t)
        for k, t in self.cbuf_replacements:
            out += struct.pack("<QI", k, t)
        if self.stage == STAGE_COMPUTE:
            out += struct.pack("<3II", *self.workgroup_size, self.shared_memory_size)
        else:
            out += self.sph
            if self.stage == STAGE_GEOMETRY:
                out += struct.pack("<8I", *self.gp_passthrough_mask)
            if with_mesh_fields and self.stage in (STAGE_TASK, STAGE_MESH):
                out += struct.pack("<3I", *self.mesh_registers) + self.mesh_primitive_header
        return bytes(out)


@dataclass
class Record:
    index: int
    offset: int
    end: int
    envs: list
    key: bytes

    @property
    def is_compute(self) -> bool:
        return self.envs[0].stage == STAGE_COMPUTE

    def serialize(self, with_mesh_fields: bool = True) -> bytes:
        return (struct.pack("<I", len(self.envs)) +
                b"".join(e.serialize(with_mesh_fields) for e in self.envs) + self.key)


class CacheError(ValueError):
    pass


# ---------------------------------------------------------------- keys
def fps_size(state: bytes) -> int:
    """FixedPipelineState::Size()"""
    raw1 = int.from_bytes(state[0:4], "little")
    xfb = (raw1 >> 6) & 1
    dvi = (raw1 >> 5) & 1
    eds3b = (raw1 >> 3) & 1
    eds = raw1 & 1
    if xfb:
        return FPS_SIZE
    if dvi and eds3b:
        return FPS_OFF["attachments"]
    if dvi:
        return FPS_OFF["attributes"]
    if eds:
        return FPS_OFF["vertex_strides"]
    return FPS_OFF["xfb_state"]


def graphics_key_hash(key: bytes) -> int:
    """GraphicsPipelineCacheKey::Hash() = CityHash64(this, 48 + state.Size())"""
    return cityhash64(key[:48 + fps_size(key[48:])])


def compute_key_hash(key: bytes) -> int:
    return cityhash64(key[:COMPUTE_KEY_SIZE])


def decode_graphics_key(key: bytes) -> dict:
    uh = struct.unpack_from("<6Q", key, 0)
    st = key[48:]
    raw1, raw2 = struct.unpack_from("<II", st, 0)
    d = {
        "unique_hashes": uh,
        "raw1": raw1, "raw2": raw2,
        "extended_dynamic_state": raw1 & 1,
        "extended_dynamic_state_2": (raw1 >> 1) & 1,
        "extended_dynamic_state_2_logic_op": (raw1 >> 2) & 1,
        "extended_dynamic_state_3_blend": (raw1 >> 3) & 1,
        "extended_dynamic_state_3_enables": (raw1 >> 4) & 1,
        "dynamic_vertex_input": (raw1 >> 5) & 1,
        "xfb_enabled": (raw1 >> 6) & 1,
        "ndc_minus_one_to_one": (raw1 >> 7) & 1,
        "polygon_mode": (raw1 >> 8) & 3,
        "tessellation_primitive": (raw1 >> 10) & 3,
        "patch_control_points_minus_one": (raw1 >> 15) & 31,
        "color_write_enable_dynamic": (raw1 >> 20) & 1,
        "attachment0_dual_source_blend": (raw1 >> 21) & 1,
        "raw1_bits_22_23": (raw1 >> 22) & 3,
        "topology": (raw1 >> 24) & 15,
        "msaa_mode": (raw1 >> 28) & 15,
        "alpha_test_func": (raw2 >> 1) & 7,
        "early_z": (raw2 >> 4) & 1,
        "depth_enabled": (raw2 >> 5) & 1,
        "depth_format": (raw2 >> 6) & 31,
        "y_negate": (raw2 >> 11) & 1,
        "provoking_vertex_last": (raw2 >> 12) & 1,
        "app_stage": (raw2 >> 17) & 7,
        "color_formats": tuple(st[8:16]),
        "driver_id": int.from_bytes(st[16:20], "little"),
        "driver_version": int.from_bytes(st[20:24], "little"),
        "alpha_test_ref": int.from_bytes(st[24:28], "little"),
        "point_size": int.from_bytes(st[28:32], "little"),
        "dynamic_state_raw1": int.from_bytes(st[72:76], "little"),
        "dynamic_state_raw2": int.from_bytes(st[76:80], "little"),
        "state_size": fps_size(st),
    }
    return d


def decode_compute_key(key: bytes) -> dict:
    uh, smem, x, y, z = struct.unpack_from("<QI3I", key, 0)
    return {"unique_hash": uh, "shared_memory_size": smem, "workgroup_size": (x, y, z)}


# ---------------------------------------------------------------- parsing
def parse_env(mv: memoryview, cur: int, n: int, mesh_fields: bool) -> Env:
    begin = cur
    if cur + ENV_HDR.size > n:
        raise CacheError(f"truncated env header at {cur}")
    (code_size, ntt, ntf, ncb, nsp, ncr, lms, tb, sa, lo, hi, vts, stage) = ENV_HDR.unpack_from(mv, cur)
    cur += ENV_HDR.size
    if stage >= len(STAGES):
        raise CacheError(f"bad stage {stage} at {begin}")
    if code_size > (64 << 20) or max(ntt, ntf, ncb, nsp, ncr) > 1_000_000:
        raise CacheError(f"implausible sizes at {begin}")
    if cur + code_size > n:
        raise CacheError(f"truncated code at {begin}")
    code = bytes(mv[cur:cur + code_size])
    cur += code_size

    def take(count, fmt):
        nonlocal cur
        s = struct.Struct("<" + fmt)
        if cur + count * s.size > n:
            raise CacheError(f"truncated map at {cur}")
        out = [s.unpack_from(mv, cur + i * s.size) for i in range(count)]
        cur += count * s.size
        return out

    tt = take(ntt, "II")
    tf = take(ntf, "Ii")
    cb = take(ncb, "QI")
    sp = take(nsp, "QB")
    cr = take(ncr, "QI")
    env = Env(begin, 0, code_size, lms, tb, sa, lo, hi, vts, stage, code, tt, tf, cb, sp, cr)
    if stage == STAGE_COMPUTE:
        if cur + 16 > n:
            raise CacheError(f"truncated compute tail at {cur}")
        x, y, z, smem = struct.unpack_from("<4I", mv, cur)
        env.workgroup_size, env.shared_memory_size = (x, y, z), smem
        cur += 16
    else:
        if cur + SPH_SIZE > n:
            raise CacheError(f"truncated sph at {cur}")
        env.sph = bytes(mv[cur:cur + SPH_SIZE])
        cur += SPH_SIZE
        if stage == STAGE_GEOMETRY:
            env.gp_passthrough_mask = struct.unpack_from("<8I", mv, cur)
            cur += 32
        if mesh_fields and stage in (STAGE_TASK, STAGE_MESH):
            env.mesh_registers = struct.unpack_from("<3I", mv, cur)
            cur += 12
            env.mesh_primitive_header = bytes(mv[cur:cur + SPH_SIZE])
            cur += SPH_SIZE
    if cur > n:
        raise CacheError(f"truncated env tail at {begin}")
    env.end = cur
    return env


def parse_cache(data: bytes, mesh_fields: bool | None = None,
                graphics_key_size: int = GRAPHICS_KEY_SIZE) -> tuple[int, Iterator[Record]]:
    """Returns (version, iterator of Records). Raises CacheError on any boundary problem."""
    if len(data) < 12 or data[:8] != MAGIC:
        raise CacheError("bad magic")
    version = struct.unpack_from("<I", data, 8)[0]
    if mesh_fields is None:
        # Fritchii's v52 records carry them too (stored since 3a2c8980ad); upstream's from v53.
        mesh_fields = version >= 52
    mv = memoryview(data)
    n = len(data)

    def gen():
        cur = 12
        idx = 0
        while cur != n:
            begin = cur
            if cur + 4 > n:
                raise CacheError(f"truncated num_envs at {cur}")
            num = struct.unpack_from("<I", mv, cur)[0]
            cur += 4
            if not 1 <= num <= 5:
                raise CacheError(f"record {idx}: bad num_envs {num} at {begin}")
            envs = []
            for _ in range(num):
                e = parse_env(mv, cur, n, mesh_fields)
                cur = e.end
                envs.append(e)
            ks = COMPUTE_KEY_SIZE if envs[0].stage == STAGE_COMPUTE else graphics_key_size
            if cur + ks > n:
                raise CacheError(f"record {idx}: truncated key at {cur}")
            key = bytes(mv[cur:cur + ks])
            cur += ks
            yield Record(idx, begin, cur, envs, key)
            idx += 1

    return version, gen()


def parse_used(data: bytes) -> tuple[list, int]:
    """Returns ([(hash, age)], trailing_bytes). Mirrors LoadUsedPipelines (stops on short read)."""
    if len(data) < 8:
        return [], len(data)
    magic, count = struct.unpack_from("<II", data, 0)
    if magic != USED_MAGIC:
        raise CacheError(f"bad used magic {magic:#x}")
    out = []
    cur = 8
    for _ in range(count):
        if cur + 9 > len(data):
            break
        h, age = struct.unpack_from("<QB", data, cur)
        out.append((h, age))
        cur += 9
    return out, len(data) - cur


def write_used(entries) -> bytes:
    out = bytearray(struct.pack("<II", USED_MAGIC, len(entries)))
    for h, age in entries:
        out += struct.pack("<QB", h, age)
    return bytes(out)


def write_cache_header(version: int) -> bytes:
    return MAGIC + struct.pack("<I", version)


def lowered_instructions(code: bytes) -> bytes | None:
    """Strip the 80-byte header, Maxwell scheduling words, padding and the native address map
    (mirrors tools/nx2/shader_sweep.py unpack_native_code) to compare programs across caches."""
    if len(code) < 96 or len(code) % 8:
        return None
    words = struct.unpack("<%dQ" % (len(code) // 8), code)
    if words[-1] != NATIVE_MAP_MAGIC:
        return None
    count = words[-2]
    if not 0 < count <= 65536 or count + 2 >= len(words):
        return None
    end = len(words) - count - 3
    return b"".join(struct.pack("<Q", words[i]) for i in range(10, end) if i % 4)


# ---------------------------------------------------------------- load-acceptance checks (mirror vk_pipeline_cache.cpp load_graphics)
# Feature bits this machine's records carry (RTX 3080, driver 621.01, DynamicState=2, VIDS on):
# raw1 low bits 0x27 (EDS1, EDS2, EDS2 logic op, dynamic vertex input), bit20 color_write_enable_dynamic,
# provoking_vertex_last=1, driver_id=4 (VK_DRIVER_ID_NVIDIA_PROPRIETARY), driver_version=0x9B404000 (621.1).
THIS_MACHINE = {"extended_dynamic_state": 1, "extended_dynamic_state_2": 1,
                "extended_dynamic_state_2_logic_op": 1, "extended_dynamic_state_3_blend": 0,
                "extended_dynamic_state_3_enables": 0, "color_write_enable_dynamic": 1,
                "dynamic_vertex_input": 1, "driver_id": 4, "driver_version": 2604679168}


def check_loadable(rec: Record, machine: dict = THIS_MACHINE) -> list[str]:
    """Problems that would make LoadPipelines skip, crash on, or mis-key this record. Empty = OK.
    (Structural parse problems raise CacheError earlier and would make Lichii DELETE the file.)"""
    p = []
    if not rec.envs:
        return ["num_envs == 0 (envs.front() is UB in LoadPipelines)"]
    if rec.is_compute:
        if len(rec.envs) != 1:
            p.append("compute record with more than one env (extra envs silently ignored)")
        ck = decode_compute_key(rec.key)
        e = rec.envs[0]
        if ck["unique_hash"] != native_shader_hash(e.code):
            p.append("compute unique_hash != NativeShaderHash(code): never matched at runtime")
        if ck["workgroup_size"] != e.workgroup_size or ck["shared_memory_size"] != e.shared_memory_size:
            p.append("compute key workgroup/shared size disagrees with env")
    else:
        if len(rec.key) != GRAPHICS_KEY_SIZE:
            p.append("graphics key size != 1056")
        gk = decode_graphics_key(rec.key)
        nz = [i for i, h in enumerate(gk["unique_hashes"]) if h]
        if len(nz) != len(rec.envs):
            p.append("num_envs != nonzero unique_hashes (out-of-bounds env access in CreateGraphicsPipeline)")
        for slot, e in zip(nz, rec.envs):
            if e.stage not in SLOT_STAGES[slot]:
                p.append(f"slot {slot} holds stage {STAGES[e.stage]}")
            if e.stage == STAGE_COMPUTE:
                p.append("compute env inside graphics record")
            if not e.has_native_map:
                p.append(f"slot {slot}: no native address map (treated as a non-native Maxwell program)")
        for f in ("extended_dynamic_state", "extended_dynamic_state_2", "extended_dynamic_state_2_logic_op",
                  "extended_dynamic_state_3_blend", "extended_dynamic_state_3_enables",
                  "color_write_enable_dynamic", "dynamic_vertex_input"):
            if gk[f] != machine[f]:
                p.append(f"{f}={gk[f]} != device {machine[f]}: silently skipped at load")
        if gk["driver_id"] != machine["driver_id"] or gk["driver_version"] != machine["driver_version"]:
            p.append("driver id/version differs: loads and builds, but runtime keys never match it")
        if any(rec.key[48 + gk["state_size"]:]):
            p.append("nonzero bytes beyond FixedPipelineState::Size() (ignored by hash/==, but not what Lichii writes)")
        mesh_env = next((e for e in rec.envs if e.stage == STAGE_MESH), None)
        for slot, e in zip(nz, rec.envs):
            h = native_shader_hash(e.code)
            if e.stage in (STAGE_TASK, STAGE_MESH) and mesh_env is not None:
                h ^= cityhash64(struct.pack("<3I", *e.mesh_registers) + mesh_env.mesh_primitive_header) | 1
            if h != gk["unique_hashes"][slot]:
                p.append(f"slot {slot}: unique hash does not match code (never matched at runtime)")
    return p


def merge_caches(base: bytes, extra: bytes, version: int = 53, extra_mesh_fields: bool | None = None,
                 machine: dict = THIS_MACHINE):
    """Return (merged_bytes, added_graphics_key_hashes, report). Base records are kept verbatim and first
    (Lichii keeps the first record per graphics key hash); extra records whose key hash is new and that pass
    check_loadable are appended re-serialized in the v53 layout. Never writes files."""
    bver, bit = parse_cache(base)
    seen_g, seen_c = set(), set()
    for r in bit:
        (seen_c if r.is_compute else seen_g).add(compute_key_hash(r.key) if r.is_compute else graphics_key_hash(r.key))
    out = bytearray(write_cache_header(version)) + base[12:]
    _, eit = parse_cache(extra, mesh_fields=extra_mesh_fields)
    added_g, report = [], {"added_graphics": 0, "added_compute": 0, "dup": 0, "rejected": {}}
    for r in eit:
        h = compute_key_hash(r.key) if r.is_compute else graphics_key_hash(r.key)
        if h in (seen_c if r.is_compute else seen_g):
            report["dup"] += 1
            continue
        probs = check_loadable(r, machine)
        if probs:
            for q in probs:
                report["rejected"][q] = report["rejected"].get(q, 0) + 1
            continue
        out += r.serialize(with_mesh_fields=True)
        if r.is_compute:
            seen_c.add(h)
            report["added_compute"] += 1
        else:
            seen_g.add(h)
            added_g.append(h)
            report["added_graphics"] += 1
    # self-check: the result must parse to EOF
    _, mit = parse_cache(bytes(out))
    report["merged_records"] = sum(1 for _ in mit)
    report["merged_bytes"] = len(out)
    return bytes(out), added_g, report
