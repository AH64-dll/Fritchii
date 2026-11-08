// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/file_sys/nx2_shader_extract.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>
#include <openssl/evp.h>
#include <zstd.h>

#include "core/file_sys/vfs/vfs.h"

namespace FileSys::Nx2ShaderExtract {
namespace {

constexpr size_t MaxBlob = size_t{1} << 30;       // largest file or inflated member read whole
constexpr size_t PeekBytes = 64 * 1024;           // what a compressed file shows before inflating
constexpr int MaxDepth = 12;
constexpr std::array<u8, 4> ControlMagic{0x34, 0x12, 0x76, 0x98};
constexpr std::array<u8, 8> BnshMagic{'B', 'N', 'S', 'H', 0, 0, 0, 0};
constexpr u32 KeyControl = 0x48;                  // control bytes in a fingerprint

struct Bad : std::runtime_error {
    using std::runtime_error::runtime_error;
};

using Bytes = std::span<const u8>;

template <typename T>
T Load(Bytes data, size_t offset) {
    if (offset > data.size() || sizeof(T) > data.size() - offset) {
        throw Bad(fmt::format("read past the end at {:#x}", offset));
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

template <typename T>
T LoadBE(Bytes data, size_t offset) {
    T value = Load<T>(data, offset);
    if constexpr (sizeof(T) == 2) {
        return static_cast<T>((value >> 8) | (value << 8));
    } else {
        u32 v = static_cast<u32>(value);
        return static_cast<T>((v >> 24) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | (v << 24));
    }
}

Bytes Sub(Bytes data, size_t offset, size_t size) {
    if (offset > data.size() || size > data.size() - offset) {
        throw Bad(fmt::format("range {:#x}+{:#x} past the end", offset, size));
    }
    return data.subspan(offset, size);
}

bool Starts(Bytes data, std::string_view magic) {
    return data.size() >= magic.size() && std::memcmp(data.data(), magic.data(), magic.size()) == 0;
}

/// An NVN2 control section: the control magic, version 0xc9.
bool IsNvn2Control(Bytes control) {
    return control.size() >= KeyControl &&
           std::memcmp(control.data(), ControlMagic.data(), 4) == 0 && control[4] == 0xc9 &&
           control[5] == 0 && control[6] == 0 && control[7] == 0;
}

/// Compute, task and mesh by the control section's stage byte (+0x56), graphics stages by the
/// program header at the code area's start (bits 10..13 of its first word: 1 vertex .. 5 pixel).
std::optional<Stage> Nvn2Stage(Bytes control, Bytes code) {
    if (control.size() > 0x56) {
        switch (control[0x56]) {
        case 5:
            return Stage::Compute;
        case 6:
            return Stage::Task;
        case 7:
            return Stage::Mesh;
        default:
            break;
        }
    }
    if (code.size() < 4) {
        return std::nullopt;
    }
    switch ((Load<u32>(code, 0) >> 10) & 0xF) {
    case 1:
        return Stage::Vertex;
    case 2:
        return Stage::TessControl;
    case 3:
        return Stage::TessEval;
    case 4:
        return Stage::Geometry;
    case 5:
        return Stage::Fragment;
    default:
        return std::nullopt;
    }
}

/// The control section names its code area's length at +0x44.
bool Consistent(Bytes control, Bytes code) {
    return IsNvn2Control(control) && Load<u32>(control, 0x44) == code.size();
}

class Zstd {
public:
    ~Zstd() {
        for (auto& [id, dict] : dictionaries) {
            ZSTD_freeDDict(dict);
        }
    }

    /// The game's dictionaries (Pack/ZsDic.pack.zs: a zstd SARC of .zsdic members).
    void AddDictionary(Bytes dictionary) {
        const unsigned id = ZSTD_getDictID_fromDict(dictionary.data(), dictionary.size());
        if (id == 0 || dictionaries.contains(id)) {
            return;
        }
        if (ZSTD_DDict* dict = ZSTD_createDDict(dictionary.data(), dictionary.size())) {
            dictionaries.emplace(id, dict);
        }
    }

    const ZSTD_DDict* Dictionary(Bytes frame) const {
        const unsigned id = ZSTD_getDictID_fromFrame(frame.data(), frame.size());
        if (id == 0) {
            return nullptr;
        }
        const auto it = dictionaries.find(id);
        if (it == dictionaries.end()) {
            throw Bad(fmt::format("zstd dictionary {} is missing", id));
        }
        return it->second;
    }

    size_t Count() const {
        return dictionaries.size();
    }

private:
    std::unordered_map<unsigned, ZSTD_DDict*> dictionaries;
};

bool IsZstd(Bytes data) {
    return data.size() >= 4 && Load<u32>(data, 0) == 0xFD2FB528;
}

/// Inflates `frame`, or only its first `limit` bytes. Per thread: the context is reused.
std::vector<u8> Inflate(const Zstd& zstd, ZSTD_DCtx* context, Bytes frame, size_t limit) {
    ZSTD_DCtx_reset(context, ZSTD_reset_session_and_parameters);
    if (const ZSTD_DDict* dict = zstd.Dictionary(frame)) {
        ZSTD_DCtx_refDDict(context, dict);
    }
    const unsigned long long declared = ZSTD_getFrameContentSize(frame.data(), frame.size());
    size_t reserve = limit;
    if (declared != ZSTD_CONTENTSIZE_UNKNOWN && declared != ZSTD_CONTENTSIZE_ERROR) {
        if (declared > MaxBlob && limit > MaxBlob) {
            throw Bad(fmt::format("zstd content of {} bytes is too large", declared));
        }
        reserve = std::min<size_t>(limit, static_cast<size_t>(declared));
    }
    std::vector<u8> out(std::min(reserve, MaxBlob));
    ZSTD_inBuffer in{frame.data(), frame.size(), 0};
    size_t produced = 0;
    while (produced < limit) {
        if (produced == out.size()) {
            if (out.size() >= MaxBlob) {
                throw Bad("zstd content is too large");
            }
            out.resize(std::min({std::max<size_t>(out.size() * 2, 1 << 16), limit, MaxBlob}));
        }
        ZSTD_outBuffer o{out.data() + produced, out.size() - produced, 0};
        const size_t status = ZSTD_decompressStream(context, &o, &in);
        if (ZSTD_isError(status)) {
            throw Bad(fmt::format("zstd: {}", ZSTD_getErrorName(status)));
        }
        produced += o.pos;
        if (status == 0 || (in.pos == in.size && o.pos == 0)) {
            break; // end of frame, or no more input
        }
    }
    out.resize(produced);
    return out;
}

std::vector<u8> Yaz0(Bytes data) {
    const u32 size = LoadBE<u32>(data, 4);
    if (size > MaxBlob) {
        throw Bad("Yaz0 content is too large");
    }
    std::vector<u8> out(size);
    size_t in = 16, o = 0;
    while (o < size) {
        const u8 group = Load<u8>(data, in++);
        for (int bit = 7; bit >= 0 && o < size; --bit) {
            if ((group >> bit) & 1) {
                out[o++] = Load<u8>(data, in++);
                continue;
            }
            const u8 b1 = Load<u8>(data, in++);
            const u8 b2 = Load<u8>(data, in++);
            const size_t distance = (size_t{b1 & 0xFu} << 8 | b2) + 1;
            size_t count = b1 >> 4;
            count = count == 0 ? size_t{Load<u8>(data, in++)} + 0x12 : count + 2;
            if (distance > o) {
                throw Bad("Yaz0 reference before the start");
            }
            for (size_t i = 0; i < count && o < size; ++i, ++o) {
                out[o] = out[o - distance];
            }
        }
    }
    return out;
}

struct Member {
    std::string name;
    Bytes data;
};

std::vector<Member> Sarc(Bytes data) {
    const bool big = Load<u16>(data, 6) == 0xFFFE; // bytes FE FF
    const auto u16_at = [&](size_t at) { return big ? LoadBE<u16>(data, at) : Load<u16>(data, at); };
    const auto u32_at = [&](size_t at) { return big ? LoadBE<u32>(data, at) : Load<u32>(data, at); };
    const u16 header = u16_at(4);
    const u32 data_offset = u32_at(12);
    if (header < 0x14 || data_offset > data.size()) {
        throw Bad("bad SARC header");
    }
    const size_t sfat = header;
    if (!Starts(Sub(data, sfat, 4), "SFAT")) {
        throw Bad("bad SFAT");
    }
    const u16 count = u16_at(sfat + 6);
    const size_t nodes = sfat + u16_at(sfat + 4);
    const size_t sfnt = nodes + size_t{count} * 0x10;
    if (!Starts(Sub(data, sfnt, 4), "SFNT")) {
        throw Bad("bad SFNT");
    }
    const size_t names = sfnt + u16_at(sfnt + 4);
    std::vector<Member> members;
    members.reserve(count);
    for (u16 i = 0; i < count; ++i) {
        const size_t node = nodes + size_t{i} * 0x10;
        const u32 flags = u32_at(node + 4);
        const u32 start = u32_at(node + 8);
        const u32 end = u32_at(node + 12);
        std::string name = fmt::format("{:08x}", u32_at(node));
        if ((flags >> 24) != 0) {
            const size_t at = names + size_t{flags & 0xFFFFFF} * 4;
            const auto tail = Sub(data, at, std::min<size_t>(data_offset, data.size()) - at);
            const auto zero = std::find(tail.begin(), tail.end(), u8{0});
            name.assign(tail.begin(), zero);
        }
        if (end < start) {
            throw Bad("bad SARC member range");
        }
        members.push_back({std::move(name), Sub(data, size_t{data_offset} + start, end - start)});
    }
    return members;
}

/// One extraction worker: its zstd context and the place it reports to.
class Walker {
public:
    using Found = std::function<void(Stage, Bytes control, Bytes code, std::string origin,
                                     std::string variation)>;

    Walker(const Zstd& zstd_, Found found_, std::vector<std::string>& errors_, Stats& stats_)
        : zstd{zstd_}, found{std::move(found_)}, errors{errors_}, stats{stats_},
          context{ZSTD_createDCtx()} {}

    ~Walker() {
        ZSTD_freeDCtx(context);
    }

    void File(const VirtualFile& file, const std::string& path) {
        const size_t size = file->GetSize();
        ++stats.files;
        if (size > MaxBlob) {
            // Large media and data: only a container at the start would be read whole.
            const auto head = file->ReadBytes(PeekBytes, 0);
            stats.file_bytes += head.size();
            if (!LooksLikeContainer(head)) {
                return;
            }
            errors.push_back(path + ": container larger than 1 GiB was not read");
            return;
        }
        const std::vector<u8> data = file->ReadAllBytes();
        stats.file_bytes += data.size();
        Guard(path, [&] { Top(data, path); });
    }

private:
    template <typename F>
    void Guard(const std::string& where, F&& f) {
        try {
            f();
        } catch (const Bad& e) {
            errors.push_back(where + ": " + e.what());
        }
    }

    static bool LooksLikeContainer(Bytes head) {
        if (head.size() < 8) {
            return false;
        }
        for (const std::string_view magic :
             {"SARC", "Yaz0", "BNSH", "FSHA", "BAHS", "NDSR", "NDVE", "YB", "BY"}) {
            if (Starts(head, magic)) {
                return true;
            }
        }
        return IsZstd(head) ||
               std::search(head.begin(), head.end(), BnshMagic.begin(), BnshMagic.end()) !=
                   head.end();
    }

    /// A RomFS file: a compressed one is inflated only if its start shows a container.
    void Top(Bytes data, const std::string& label) {
        if (IsZstd(data)) {
            const auto head = Inflate(zstd, context, data, PeekBytes);
            if (!LooksLikeContainer(head) || IsZstd(head)) {
                ++stats.peeked_only;
                return;
            }
            const auto inner = Inflate(zstd, context, data, MaxBlob);
            stats.inflated_bytes += inner.size();
            Walk(inner, label, 1);
            return;
        }
        Walk(data, label, 0);
    }

    void Walk(Bytes data, const std::string& label, int depth) {
        if (depth > MaxDepth || data.size() < 8) {
            return;
        }
        if (IsZstd(data)) {
            const auto inner = Inflate(zstd, context, data, MaxBlob);
            stats.inflated_bytes += inner.size();
            Walk(inner, label, depth + 1);
        } else if (Starts(data, "Yaz0")) {
            const auto inner = Yaz0(data);
            stats.inflated_bytes += inner.size();
            Walk(inner, label, depth + 1);
        } else if (Starts(data, "SARC")) {
            ++stats.containers;
            for (const auto& member : Sarc(data)) {
                const std::string where = label + "/" + member.name;
                Guard(where, [&] { Walk(member.data, where, depth + 1); });
            }
        } else if (Starts(data, "BAHS")) {
            ++stats.containers;
            Bahs(data, label);
        } else if (Starts(data, "NDSR")) {
            ++stats.containers;
            Ndsr(data, label);
        } else if (Starts(data, "NDVE")) {
            ++stats.containers;
            Ndve(data, label);
        } else {
            // BNSH files, and BNSH inside other data (BFSHA archives, effect bundles, layouts).
            if (!EmbeddedBnsh(data, label)) {
                SeadPrimitive(data, label);
            }
        }
    }

    void Emit(Stage stage, Bytes control, Bytes code, std::string origin, std::string variation) {
        ++stats.occurrences;
        found(stage, control, code, std::move(origin), std::move(variation));
    }

    bool EmbeddedBnsh(Bytes data, const std::string& label) {
        bool any = false;
        auto it = data.begin();
        while ((it = std::search(it, data.end(), BnshMagic.begin(), BnshMagic.end())) !=
               data.end()) {
            const size_t base = static_cast<size_t>(it - data.begin());
            const size_t size =
                base + 0x20 <= data.size() ? size_t{Load<u32>(data, base + 0x1c)} : 0;
            const bool bounded = size >= 0x98 && size <= data.size() - base;
            if (bounded) {
                Guard(fmt::format("{}#BNSH@{:#x}", label, base), [&] {
                    Bnsh(Sub(data, base, size), label, base);
                    any = true;
                    ++stats.containers;
                });
            }
            it += static_cast<std::ptrdiff_t>(bounded ? size : 8);
        }
        return any;
    }

    // BNSH: +0x60 'grsc', +0x7c variation count, +0x80 variation table; a variation is 0x40
    // bytes with its program at +0x10; a program has six stage pointers from +0x08 (vertex,
    // hull, domain, geometry, pixel, compute) and NVN2's task and mesh at +0x40 and +0x48; a
    // stage record has its control at +0x08, code at +0x10, code length +0x18, control length
    // +0x1c. All pointers count from the BNSH start.
    void Bnsh(Bytes d, const std::string& label, size_t base) {
        if (!Starts(Sub(d, 0x60, 4), "grsc")) {
            throw Bad("no grsc block");
        }
        const u32 count = Load<u32>(d, 0x7c);
        const u64 table = Load<u64>(d, 0x80);
        static constexpr std::array slots{Stage::Vertex,   Stage::TessControl, Stage::TessEval,
                                          Stage::Geometry, Stage::Fragment,    Stage::Compute};
        for (u32 v = 0; v < count; ++v) {
            const u64 program = Load<u64>(d, table + 0x40 * u64{v} + 0x10);
            if (program == 0) {
                continue;
            }
            const std::string variation = fmt::format("{}#BNSH@{:#x}#var{}", label, base, v);
            const auto stage_at = [&](u64 record, Stage stage, bool optional) {
                const u64 control_at = Load<u64>(d, record + 0x08);
                const u64 code_at = Load<u64>(d, record + 0x10);
                const u32 code_size = Load<u32>(d, record + 0x18);
                const u32 control_size = Load<u32>(d, record + 0x1c);
                if (optional && (control_size < KeyControl || control_size >= 0x10000 ||
                                 control_at + control_size > d.size() ||
                                 code_at + code_size > d.size())) {
                    return;
                }
                const auto control = Sub(d, control_at, control_size);
                const auto code = Sub(d, code_at, code_size);
                if (optional) {
                    const u8 expected = stage == Stage::Task ? 6 : 7;
                    if (!IsNvn2Control(control) || control.size() <= 0x56 ||
                        control[0x56] != expected) {
                        return;
                    }
                }
                if (!Consistent(control, code)) {
                    throw Bad(fmt::format("variation {} {}: control +0x44 does not match the code",
                                          v, StageName(stage)));
                }
                Emit(stage, control, code, variation, variation);
            };
            for (size_t slot = 0; slot < slots.size(); ++slot) {
                const u64 record = Load<u64>(d, program + 8 + 8 * slot);
                if (record != 0) {
                    stage_at(record, slots[slot], false);
                }
            }
            for (const auto& [offset, stage] :
                 {std::pair{u64{0x40}, Stage::Task}, std::pair{u64{0x48}, Stage::Mesh}}) {
                if (program + offset + 8 > d.size()) {
                    continue;
                }
                const u64 record = Load<u64>(d, program + offset);
                if (record != 0 && record + 0x20 <= d.size()) {
                    stage_at(record, stage, true);
                }
            }
        }
    }

    // agl's BAHS (sharcb): +0x18 code region length, +0x1c code region start. A program entry
    // in the area before it is u32 code offset (multiple of 0x100), u32 0, u32 control length,
    // i32 distance to the control section. An NVN2 section's +0x38..+0x47 reads as such an
    // entry, so entries inside a found control section are dropped (nxnative: Ocarina of
    // Time's agl_common.sharcb@0x357f0).
    void Bahs(Bytes d, const std::string& label) {
        const u32 region_size = Load<u32>(d, 0x18);
        const u32 region = Load<u32>(d, 0x1c);
        if (size_t{region} + region_size > d.size()) {
            throw Bad("BAHS code region out of bounds");
        }
        struct Entry {
            size_t at, control_at;
            Stage stage;
            Bytes control, code;
        };
        std::vector<Entry> entries;
        for (size_t a = 0; a + 16 < region; a += 4) {
            const u32 offset = Load<u32>(d, a);
            const u32 zero = Load<u32>(d, a + 4);
            const u32 control_size = Load<u32>(d, a + 8);
            const s32 distance = Load<s32>(d, a + 12);
            if (zero != 0 || offset % 0x100 != 0 || control_size < KeyControl ||
                control_size >= 0x10000) {
                continue;
            }
            const s64 control_at = static_cast<s64>(a) + distance;
            if (control_at < 0 || static_cast<u64>(control_at) + control_size > region) {
                continue;
            }
            const auto control = d.subspan(static_cast<size_t>(control_at), control_size);
            if (!IsNvn2Control(control)) {
                continue;
            }
            const u32 code_size = Load<u32>(control, 0x44);
            if (u64{offset} + code_size > region_size) {
                continue;
            }
            const auto code = d.subspan(size_t{region} + offset, code_size);
            const auto stage = Nvn2Stage(control, code);
            if (!stage) {
                continue;
            }
            entries.push_back({a, static_cast<size_t>(control_at), *stage, control, code});
        }
        std::vector<const Entry*> kept;
        for (const auto& entry : entries) {
            const bool inside = std::any_of(entries.begin(), entries.end(), [&](const Entry& e) {
                return e.control_at <= entry.at && entry.at < e.control_at + KeyControl;
            });
            if (!inside) {
                kept.push_back(&entry);
            }
        }
        std::vector<size_t> taken;
        for (const Entry* entry : kept) {
            taken.push_back(entry->at);
        }
        const auto variations = BahsVariations(d, label, taken);
        for (const Entry* entry : kept) {
            const std::string where = fmt::format("{}#BAHS@{:#x}", label, entry->at);
            const auto it = variations.find(entry->at);
            Emit(entry->stage, entry->control, entry->code, where,
                 it != variations.end() ? it->second : where);
        }
    }

    // The variations of a BAHS file's programs. Right before the first entry is a binary array
    // (u32 byte size, u32 count, then per binary: u32 record size, u32 kind (0 vertex, 1
    // fragment, 3 compute), two u32, the 16-byte entry), and right after it a program array (u32
    // byte size, u32 count, then per program: u32 record size, u32 name length, u32 stage mask,
    // u32 index of its first binary, ...). A program's binaries run to the next program's first;
    // its n-th vertex and n-th fragment binary are one variation. Arrays that do not read back
    // give no variations (nxnative: bahs_variations).
    static std::unordered_map<size_t, std::string> BahsVariations(Bytes d, const std::string& label,
                                                                  const std::vector<size_t>& taken) {
        std::unordered_map<size_t, std::string> result;
        if (taken.empty()) {
            return result;
        }
        const s64 slack = std::max<s64>(1, static_cast<s64>(taken.size()) / 100);
        const size_t first = *std::min_element(taken.begin(), taken.end());
        if (first < 0x18 + 0x20) {
            return result;
        }
        const size_t array = first - 0x18;
        const u32 array_size = Load<u32>(d, array);
        const u32 array_count = Load<u32>(d, array + 4);
        if (u64{array} + array_size + 8 > d.size() ||
            static_cast<s64>(array_count) < static_cast<s64>(taken.size()) - slack) {
            return result;
        }
        struct Record {
            size_t entry;
            u32 kind;
        };
        std::vector<Record> records;
        size_t o = array + 8;
        for (u32 i = 0; i < array_count; ++i) {
            if (o + 0x20 > array + array_size) {
                return result;
            }
            const u32 size = Load<u32>(d, o);
            if (size < 0x20) {
                return result;
            }
            records.push_back({o + 0x10, Load<u32>(d, o + 4)});
            o += size;
        }
        const auto in_records = [&](size_t at) {
            return std::any_of(records.begin(), records.end(),
                               [&](const Record& r) { return r.entry == at; });
        };
        const s64 outside = std::count_if(taken.begin(), taken.end(),
                                          [&](size_t at) { return !in_records(at); });
        if (o != array + array_size || outside > slack) {
            return result;
        }
        const u32 programs_size = Load<u32>(d, o);
        const u32 programs = Load<u32>(d, o + 4);
        if (u64{o} + programs_size > d.size() || programs == 0) {
            return result;
        }
        std::vector<u32> bases;
        size_t q = o + 8;
        for (u32 p = 0; p < programs; ++p) {
            if (q + 16 > o + programs_size) {
                return result;
            }
            const u32 size = Load<u32>(d, q);
            const u32 base = Load<u32>(d, q + 12);
            if (size < 16 || base > array_count || (!bases.empty() && base < bases.back())) {
                return result;
            }
            bases.push_back(base);
            q += size;
        }
        if (q != o + programs_size) {
            return result;
        }
        bases.push_back(array_count);
        const std::unordered_set<size_t> taken_set(taken.begin(), taken.end());
        for (u32 p = 0; p < programs; ++p) {
            std::array<u32, 2> seen{};
            for (u32 i = bases[p]; i < bases[p + 1]; ++i) {
                const auto& record = records[i];
                if (record.kind > 1) {
                    continue; // compute and the other kinds pair with nothing
                }
                if (taken_set.contains(record.entry)) {
                    result.emplace(record.entry, fmt::format("{}#BAHS#prog{}#var{}", label, p,
                                                             seen[record.kind]));
                }
                ++seen[record.kind];
            }
        }
        return result;
    }

    // agl's NDSR pack: 16-byte records from +0x10 (or +0x14) up to a zero one: u32 control
    // offset, u32 control length << 4 | type, u32 code offset, u32. Records not pointing at a
    // control section hold data and may shift the next by 8. A vertex record is followed by its
    // pixel record.
    void Ndsr(Bytes d, const std::string& label) {
        const auto points_at_control = [&](size_t at) {
            return u64{at} + KeyControl <= d.size() && IsNvn2Control(d.subspan(at, KeyControl));
        };
        const auto score = [&](size_t start) {
            int n = 0;
            for (size_t i = 0; i < 16 && start + 16 * i + 16 <= d.size(); ++i) {
                n += points_at_control(Load<u32>(d, start + 16 * i)) ? 1 : 0;
            }
            return n;
        };
        size_t o = score(0x14) > score(0x10) ? 0x14 : 0x10;
        std::string vertex;
        while (o + 16 <= d.size()) {
            const u32 control_at = Load<u32>(d, o);
            const u32 control_word = Load<u32>(d, o + 4);
            const u32 code_at = Load<u32>(d, o + 8);
            const u32 last = Load<u32>(d, o + 12);
            if ((control_at | control_word | code_at | last) == 0) {
                break;
            }
            if (!points_at_control(control_at)) {
                o += 8;
                continue;
            }
            o += 16;
            const size_t control_size = std::max<size_t>(control_word >> 4, KeyControl);
            if (u64{control_at} + control_size > d.size()) {
                continue;
            }
            const auto control = d.subspan(control_at, control_size);
            const u32 code_size = Load<u32>(control, 0x44);
            if (u64{code_at} + code_size > d.size()) {
                continue;
            }
            const auto code = d.subspan(code_at, code_size);
            const auto stage = Nvn2Stage(control, code);
            if (!stage || !Consistent(control, code)) {
                continue;
            }
            const std::string where = fmt::format("{}#NDSR@{:#x}", label, control_at);
            std::string variation = where;
            if (*stage == Stage::Vertex) {
                vertex = where;
            } else if (*stage == Stage::Fragment && !vertex.empty()) {
                variation = vertex; // the pair is labelled by its vertex record
                vertex.clear();
            }
            Emit(*stage, control, code, where, variation);
        }
    }

    // nn::ndve's movie enhancement file: +0xc length of its NDSR pack, +0x14 count of 8-byte
    // model records, then the pack, then the networks: control sections each naming their own
    // length (+0x30) and their code area's (+0x44), the code area right behind.
    void Ndve(Bytes d, const std::string& label) {
        const size_t pack = 0x18 + 8 * size_t{Load<u32>(d, 0x14)};
        const size_t networks = pack + Load<u32>(d, 0xc);
        Ndsr(Sub(d, pack, networks - pack), label);
        for (size_t c = (networks + 3) & ~size_t{3}; c + KeyControl <= d.size(); c += 4) {
            if (std::memcmp(d.data() + c, ControlMagic.data(), 4) != 0 ||
                !IsNvn2Control(d.subspan(c, KeyControl))) {
                continue;
            }
            const u32 own = Load<u32>(d, c + 0x30);
            const u32 code_size = Load<u32>(d, c + 0x44);
            if (own < KeyControl || c + own + code_size > d.size()) {
                continue;
            }
            const auto control = d.subspan(c, own);
            const auto code = d.subspan(c + own, code_size);
            if (const auto stage = Nvn2Stage(control, code)) {
                const std::string where = fmt::format("{}#NDVE@{:#x}", label, c);
                Emit(*stage, control, code, where, where);
            }
        }
    }

    // sead's NVN2 primitive drawer: four u32 offsets (vertex control, pixel control, vertex
    // code, pixel code); each control section names its own length (+0x30) and its code's (+0x44).
    void SeadPrimitive(Bytes d, const std::string& label) {
        if (d.size() < 0x10 || d.size() > (1u << 20)) {
            return;
        }
        const u32 vc = Load<u32>(d, 0), pc = Load<u32>(d, 4), vk = Load<u32>(d, 8),
                  pk = Load<u32>(d, 12);
        struct Piece {
            Stage stage;
            Bytes control, code;
        };
        std::vector<Piece> pieces;
        for (const auto& [stage, c, k] : {std::tuple{Stage::Vertex, vc, vk},
                                          std::tuple{Stage::Fragment, pc, pk}}) {
            if (c < 0x10 || u64{c} + KeyControl > d.size() ||
                !IsNvn2Control(d.subspan(c, KeyControl))) {
                return;
            }
            const u32 own = std::max<u32>(Load<u32>(d, c + 0x30), KeyControl);
            if (u64{c} + own > d.size()) {
                return;
            }
            const auto control = d.subspan(c, own);
            const u32 n = Load<u32>(control, 0x44);
            if (u64{k} + n > d.size()) {
                return;
            }
            const auto code = d.subspan(k, n);
            if (Nvn2Stage(control, code) != stage) {
                return;
            }
            pieces.push_back({stage, control, code});
        }
        ++stats.containers;
        const std::string variation = fmt::format("{}#SEADPRIM", label);
        for (const auto& piece : pieces) {
            Emit(piece.stage, piece.control, piece.code, variation, variation);
        }
    }

    const Zstd& zstd;
    Found found;
    std::vector<std::string>& errors;
    Stats& stats;
    ZSTD_DCtx* context;
};

struct FileItem {
    VirtualFile file;
    std::string path;
};

void Collect(const VirtualDir& dir, const std::string& path, std::vector<FileItem>& out) {
    for (const auto& file : dir->GetFiles()) {
        out.push_back({file, path + "/" + file->GetName()});
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        Collect(sub, path + "/" + sub->GetName(), out);
    }
}

void LoadDictionaries(const std::span<const VirtualDir> roots, Zstd& zstd) {
    ZSTD_DCtx* context = ZSTD_createDCtx();
    for (const auto& root : roots) {
        const auto pack = root ? root->GetDirectoryRelative("Pack") : nullptr;
        const auto file = pack ? pack->GetFile("ZsDic.pack.zs") : nullptr;
        if (!file) {
            continue;
        }
        try {
            const auto bytes = file->ReadAllBytes();
            const auto sarc = Inflate(zstd, context, bytes, MaxBlob);
            for (const auto& member : Sarc(sarc)) {
                if (member.data.size() >= 4 && Load<u32>(member.data, 0) == 0xEC30A437) {
                    zstd.AddDictionary(member.data);
                }
            }
        } catch (const Bad&) {
        }
    }
    ZSTD_freeDCtx(context);
}

} // Anonymous namespace

const char* StageName(Stage stage) noexcept {
    switch (stage) {
    case Stage::Vertex:
        return "vertex";
    case Stage::TessControl:
        return "hull";
    case Stage::TessEval:
        return "domain";
    case Stage::Geometry:
        return "geometry";
    case Stage::Fragment:
        return "pixel";
    case Stage::Compute:
        return "compute";
    case Stage::Task:
        return "task";
    case Stage::Mesh:
        return "mesh";
    }
    return "unknown";
}

u64 Fingerprint(std::span<const u8> control, std::span<const u8> code) {
    std::array<u8, KeyControl> key{};
    std::memcpy(key.data(), control.data(), std::min<size_t>(control.size(), KeyControl));
    std::memset(key.data() + 0x30, 0, 4);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    std::array<u8, 32> digest{};
    unsigned length = 0;
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx.get(), key.data(), key.size()) != 1 ||
        EVP_DigestUpdate(ctx.get(), code.data(), code.size()) != 1 ||
        EVP_DigestFinal_ex(ctx.get(), digest.data(), &length) != 1) {
        throw std::runtime_error("SHA-256 failed");
    }
    u64 value = 0;
    for (int i = 0; i < 8; ++i) {
        value = value << 8 | digest[i]; // the first 16 hex digits, as nxnative prints them
    }
    return value;
}

Result Extract(std::span<const VirtualDir> roots, unsigned threads, Progress* progress,
               std::stop_token stop) {
    Zstd zstd;
    LoadDictionaries(roots, zstd);
    std::vector<FileItem> files;
    for (const auto& root : roots) {
        if (root) {
            Collect(root, "", files);
        }
    }
    if (progress) {
        progress->files_total = files.size();
    }

    Result result;
    std::mutex mutex;
    struct Seen {
        u32 index;
        size_t file; ///< the earliest file it was found in decides its origin
    };
    std::unordered_map<u64, Seen> seen;
    std::unordered_map<std::string, size_t> variation_index;

    const unsigned count =
        threads ? threads : std::max(1u, std::thread::hardware_concurrency());
    std::atomic<size_t> next{0};
    std::vector<std::vector<std::string>> errors(count);
    std::vector<Stats> stats(count);
    const auto work = [&](unsigned worker) {
        size_t current = 0;
        Walker walker(
            zstd,
            [&](Stage stage, Bytes control, Bytes code, std::string origin, std::string variation) {
                // Only the control bytes a fingerprint covers matter to the program; keep them all
                // (the translator reads up to the section's own length).
                const u64 fp = Fingerprint(control, code);
                std::scoped_lock lock{mutex};
                auto [it, inserted] = seen.try_emplace(fp, Seen{0, current});
                if (inserted) {
                    it->second.index = static_cast<u32>(result.programs.size());
                    result.programs.push_back({stage, fp, {control.begin(), control.end()},
                                               {code.begin(), code.end()}, origin});
                } else if (current < it->second.file) {
                    it->second.file = current;
                    result.programs[it->second.index].origin = origin;
                }
                auto [v, added] = variation_index.try_emplace(variation, result.variations.size());
                if (added) {
                    result.variations.push_back({std::move(variation), {}});
                }
                auto& members = result.variations[v->second].programs;
                if (std::find(members.begin(), members.end(), it->second.index) == members.end()) {
                    members.push_back(it->second.index);
                }
            },
            errors[worker], stats[worker]);
        while (!stop.stop_requested()) {
            current = next.fetch_add(1);
            if (current >= files.size()) {
                break;
            }
            try {
                walker.File(files[current].file, files[current].path);
            } catch (const std::exception& e) {
                errors[worker].push_back(files[current].path + ": " + e.what());
            }
            if (progress) {
                progress->files_done.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    {
        std::vector<std::jthread> pool;
        for (unsigned i = 0; i < count; ++i) {
            pool.emplace_back(work, i);
        }
    }
    for (unsigned i = 0; i < count; ++i) {
        result.errors.insert(result.errors.end(), errors[i].begin(), errors[i].end());
        result.stats.files += stats[i].files;
        result.stats.file_bytes += stats[i].file_bytes;
        result.stats.inflated_bytes += stats[i].inflated_bytes;
        result.stats.peeked_only += stats[i].peeked_only;
        result.stats.containers += stats[i].containers;
        result.stats.occurrences += stats[i].occurrences;
    }
    if (zstd.Count() == 0) {
        result.errors.push_back("Pack/ZsDic.pack.zs: no zstd dictionaries (files that need one "
                                "are reported as missing their dictionary)");
    }
    return result;
}

} // namespace FileSys::Nx2ShaderExtract
