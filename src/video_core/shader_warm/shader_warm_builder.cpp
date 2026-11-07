// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/loader.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/shader_warm/shader_archives.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"

namespace VideoCommon::ShaderWarm {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view StageReading = "Reading game files";
constexpr std::string_view StageExtracting = "Extracting shaders";
constexpr std::string_view StageConverting = "Converting";
constexpr std::string_view StageWriting = "Writing";
/// Conversions in flight (each holds a copy of its program).
constexpr size_t MaxPendingJobs = 2048;

fs::path TitleDir(u64 title_id) {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir) /
           fmt::format("{:016x}", title_id);
}

bool EndsWith(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

struct ArchiveFile {
    std::string path; ///< Inside the RomFS, '/' separated (the archive identity)
    FileSys::VirtualFile file;
};

void ListFiles(const FileSys::VirtualDir& dir, const std::string& prefix,
               std::vector<ArchiveFile>& out) {
    for (const auto& file : dir->GetFiles()) {
        out.push_back({prefix + file->GetName(), file});
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        ListFiles(sub, prefix + sub->GetName() + "/", out);
    }
}

/// The game's RomFS trees in the order the offline pipeline read them: a .nup gives its base
/// and newest update package (both are read, like the exported packages 0 and 1); other
/// formats give the loader's RomFS.
struct GameFiles {
    std::unique_ptr<FileSys::NupContainer> nup;
    std::unique_ptr<Loader::AppLoader> loader;
    std::vector<FileSys::VirtualDir> romfs;
};

std::optional<std::string> OpenGame(Core::System& system, const fs::path& game_path,
                                    GameFiles& game) {
    FileSys::VirtualFilesystem filesystem = system.GetFilesystem();
    if (filesystem == nullptr) {
        filesystem = std::make_shared<FileSys::RealVfsFilesystem>();
    }
    const auto file =
        filesystem->OpenFile(Common::FS::PathToUTF8String(game_path), FileSys::OpenMode::Read);
    if (file == nullptr) {
        return "Cannot open the game file";
    }
    if (FileSys::NupContainer::IsNupFile(file)) {
        // Small memory cache: every member is read once.
        game.nup = std::make_unique<FileSys::NupContainer>(file, 128 * 1024 * 1024);
        if (game.nup->GetStatus() != Loader::ResultStatus::Success) {
            return "The game file is not a valid package";
        }
        std::vector<size_t> packages;
        if (const auto base = game.nup->FindBasePackage()) {
            packages.push_back(*base);
        }
        if (const auto update = game.nup->FindBestUpdatePackage()) {
            packages.push_back(*update);
        }
        std::sort(packages.begin(), packages.end());
        for (const size_t package : packages) {
            if (auto dir = game.nup->GetRomFSDir(package)) {
                game.romfs.push_back(std::move(dir));
            }
        }
    } else {
        game.loader = Loader::GetLoader(system, file);
        if (game.loader == nullptr) {
            return "Unsupported game file format";
        }
        FileSys::VirtualFile romfs;
        if (game.loader->ReadRomFS(romfs) != Loader::ResultStatus::Success || romfs == nullptr) {
            return "Cannot read the game's RomFS (are the keys installed?)";
        }
        if (auto dir = FileSys::ExtractRomFS(romfs)) {
            game.romfs.push_back(std::move(dir));
        }
    }
    if (game.romfs.empty()) {
        return "The game has no RomFS";
    }
    return std::nullopt;
}

struct PairHash {
    u128 a{};
    u128 b{};
    bool operator==(const PairHash&) const = default;
};
struct PairHasher {
    size_t operator()(const PairHash& h) const noexcept {
        return static_cast<size_t>(h.a[0] ^ (h.b[0] * 0x9E3779B97F4A7C15ULL));
    }
};
struct U128Hasher {
    size_t operator()(const u128& h) const noexcept {
        return static_cast<size_t>(h[0]);
    }
};

u128 Hash128(std::span<const u8> bytes) {
    return Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

/// One unique vertex or fragment program (make_graphics_warm.py: first manifest program per
/// code hash, with the origins of that program's code + control identity).
struct GraphicsProgram {
    u32 stage{};
    u128 control{};
    std::vector<u32> archives;
    bool built{};
    u64 hash{};
};

struct Job {
    size_t seq{};
    bool graphics{};
    u32 stage{};
    size_t graphics_index{};
    std::vector<u8> program;
    ComputeControl control{};
};

struct JobResult {
    bool ok{};
    std::vector<u8> bytes; ///< Compute: full entry. Graphics: env record.
    ComputeKey compute_key{};
    u64 hash{};
};

class Converter {
public:
    Converter(size_t threads, std::stop_token stop_) : stop{std::move(stop_)} {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] { Work(); });
        }
    }

    ~Converter() {
        {
            std::scoped_lock lock{mutex};
            closing = true;
        }
        jobs_cv.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

    void Push(Job job) {
        {
            std::scoped_lock lock{mutex};
            jobs.push_back(std::move(job));
            ++pushed;
        }
        jobs_cv.notify_one();
    }

    /// Next result in submission order; waits when `wait` is set, else empty when not ready.
    std::optional<JobResult> Next(bool wait) {
        std::unique_lock lock{mutex};
        for (;;) {
            const auto it = results.find(next);
            if (it != results.end()) {
                JobResult result = std::move(it->second);
                results.erase(it);
                ++next;
                return result;
            }
            if (!wait || next == pushed || stop.stop_requested()) {
                return std::nullopt;
            }
            results_cv.wait_for(lock, std::chrono::milliseconds{100});
        }
    }

    size_t Pending() {
        std::scoped_lock lock{mutex};
        return pushed - next;
    }

private:
    void Work() {
        Common::SetCurrentThreadName("ShaderWarmBuild");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        for (;;) {
            Job job;
            {
                std::unique_lock lock{mutex};
                jobs_cv.wait(lock, [this] { return closing || !jobs.empty(); });
                if (jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            JobResult result;
            if (!stop.stop_requested()) {
                result = Convert(job);
            }
            {
                std::scoped_lock lock{mutex};
                results.emplace(job.seq, std::move(result));
            }
            results_cv.notify_all();
        }
    }

    static JobResult Convert(const Job& job) {
        JobResult result;
        try {
            LoweredCode lowered;
            if (const auto error = LowerProgram(job.program, job.graphics, lowered)) {
                LOG_DEBUG(Render_Vulkan, "Shader warm-up: program not converted: {}", *error);
                return result;
            }
            if (job.graphics) {
                result.hash = StageHash(lowered.code);
                result.bytes = GraphicsEnv(job.stage, lowered, job.program);
            } else {
                result.bytes = ComputeEntry(lowered, job.control, result.compute_key);
            }
            result.ok = true;
        } catch (const std::exception& e) {
            LOG_DEBUG(Render_Vulkan, "Shader warm-up: conversion failed: {}", e.what());
        }
        return result;
    }

    std::stop_token stop;
    std::mutex mutex;
    std::condition_variable jobs_cv;
    std::condition_variable results_cv;
    std::deque<Job> jobs;
    std::map<size_t, JobResult> results;
    size_t pushed{};
    size_t next{};
    bool closing{};
    std::vector<std::thread> workers;
};

template <typename T>
void Write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void WriteHeader(std::ofstream& out) {
    out.write("yuzucach", 8);
    Write(out, WarmCacheVersion);
}

} // Anonymous namespace

fs::path ComputeWarmPath(u64 title_id) {
    return TitleDir(title_id) / "warm-compute.bin";
}

fs::path GraphicsWarmPath(u64 title_id) {
    return TitleDir(title_id) / "warm-graphics.bin";
}

fs::path WarmMarkerPath(const fs::path& warm_file) {
    fs::path marker = warm_file;
    marker.replace_extension(".done");
    return marker;
}

bool DeleteWarmFiles(u64 title_id) {
    bool ok = true;
    for (const auto& warm : {ComputeWarmPath(title_id), GraphicsWarmPath(title_id)}) {
        fs::path tmp = warm;
        tmp += ".tmp";
        for (const auto& path : {warm, WarmMarkerPath(warm), tmp}) {
            std::error_code ec;
            fs::remove(path, ec);
            ok = ok && !ec;
        }
    }
    return ok;
}

Result BuildWarmFiles(Core::System& system, const fs::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop) {
    return BuildWarmFilesWith(system, game_path, title_id, BuildOptions{}, progress,
                              std::move(stop));
}

Result BuildWarmFilesWith(Core::System& system, const fs::path& game_path, u64 title_id,
                          const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop) {
    Result result;
    const auto report = [&](std::string_view stage, size_t done, size_t total) {
        if (progress) {
            progress(Progress{std::string(stage), done, total});
        }
    };
    const auto begin = std::chrono::steady_clock::now();
    report(StageReading, 0, 0);

    const fs::path output_dir = options.output_dir.empty() ? TitleDir(title_id) : options.output_dir;
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    const fs::path compute_path = output_dir / "warm-compute.bin";
    const fs::path graphics_path = output_dir / "warm-graphics.bin";
    fs::path compute_tmp = compute_path;
    compute_tmp += ".tmp";
    fs::path graphics_tmp = graphics_path;
    graphics_tmp += ".tmp";
    std::ofstream compute_out;
    std::ofstream graphics_out;
    const auto fail = [&](std::string error) {
        compute_out.close();
        graphics_out.close();
        std::error_code ignored;
        fs::remove(compute_tmp, ignored);
        fs::remove(graphics_tmp, ignored);
        result.ok = false;
        result.error = std::move(error);
        LOG_WARNING(Render_Vulkan, "Shader warm-up build for {:016x} failed: {}", title_id,
                    result.error);
        return result;
    };

    // Draw state guesses.
    std::vector<fs::path> caches;
    if (options.runtime_caches) {
        caches = *options.runtime_caches;
    } else {
        const auto own = VideoCommon::VersionedPipelineCachePath(
            TitleDir(title_id), "vulkan", VideoCommon::VulkanPipelineCacheVersion);
        if (fs::exists(own, ec)) {
            caches.push_back(own);
        }
    }
    const RuntimeStats stats = ReadRuntimeStats(caches);
    const bool use_stats = stats.occurrences != 0;
    const GraphicsKey vertex_template = stats.templates[0] ? *stats.templates[0]
                                                           : DefaultTemplate(true);
    const GraphicsKey fragment_template = stats.templates[1] ? *stats.templates[1]
                                                             : DefaultTemplate(false);
    LOG_INFO(Render_Vulkan,
             "Shader warm-up build for {:016x}: {} runtime stage records from {} cache(s){}",
             title_id, stats.occurrences, caches.size(),
             use_stats ? "" : ", built-in draw state defaults");

    GameFiles game;
    if (const auto error = OpenGame(system, game_path, game)) {
        return fail(*error);
    }

    // Archive list, in the offline pipeline's order: every *.bfsha.zs and *.sarc.zs of each
    // package, then each package's Effect/*.vsetb.byml.zs, sorted by path inside a package.
    std::vector<ArchiveFile> archives;
    std::vector<ArchiveFile> effects;
    ZstdDictionaries dictionaries;
    FileSys::VirtualFile dictionary_pack;
    for (const auto& romfs : game.romfs) {
        std::vector<ArchiveFile> files;
        ListFiles(romfs, "", files);
        std::sort(files.begin(), files.end(),
                  [](const ArchiveFile& a, const ArchiveFile& b) { return a.path < b.path; });
        std::vector<ArchiveFile> package_effects;
        for (auto& file : files) {
            if (EndsWith(file.path, ".bfsha.zs") || EndsWith(file.path, ".sarc.zs")) {
                archives.push_back(file);
            } else if (file.path.starts_with("Effect/") &&
                       file.path.find('/', 7) == std::string::npos &&
                       EndsWith(file.path, ".vsetb.byml.zs")) {
                package_effects.push_back(file);
            } else if (file.path == "Pack/ZsDic.pack.zs") {
                dictionary_pack = file.file; // the newest package wins
            }
        }
        effects.insert(effects.end(), package_effects.begin(), package_effects.end());
    }
    archives.insert(archives.end(), effects.begin(), effects.end());
    if (dictionary_pack) {
        try {
            dictionaries.Load(dictionary_pack->ReadAllBytes());
        } catch (const ArchiveError& e) {
            LOG_WARNING(Render_Vulkan, "Shader warm-up: zstd dictionary pack: {}", e.what());
        }
    }
    if (archives.empty()) {
        return fail("No shader archives found in the game files");
    }
    LOG_INFO(Render_Vulkan, "Shader warm-up build: {} archives, {} zstd dictionaries",
             archives.size(), dictionaries.Count());

    compute_out.open(compute_tmp, std::ios::binary | std::ios::trunc);
    graphics_out.open(graphics_tmp, std::ios::binary | std::ios::trunc);
    if (!compute_out || !graphics_out) {
        return fail("Cannot write to " + Common::FS::PathToUTF8String(output_dir));
    }
    WriteHeader(compute_out);
    WriteHeader(graphics_out);

    // Bookkeeping in program order (the order the manifest lists programs).
    std::unordered_set<PairHash, PairHasher> compute_identities;
    std::unordered_map<u128, size_t, U128Hasher> graphics_index; // code -> program
    std::vector<GraphicsProgram> graphics;
    std::unordered_set<std::string> compute_keys;
    std::unordered_set<u64> fragment_hashes;
    std::unordered_set<u64> vertex_hashes;
    struct WrittenVertex {
        size_t program;
        u64 key_offset;
    };
    std::vector<WrittenVertex> written_vertices;
    struct HeldVertex {
        size_t program;
        std::vector<u8> env;
    };
    std::vector<HeldVertex> held_vertices;
    size_t archive_errors = 0;
    size_t member_errors = 0;
    size_t seq = 0;

    const size_t threads =
        options.threads != 0
            ? options.threads
            : std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 2, 6);
    auto converter = std::make_unique<Converter>(threads, stop);

    // Consumes finished conversions in submission order (first program wins on equal keys).
    std::vector<std::pair<bool, size_t>> job_kinds; // graphics?, graphics program index
    size_t consumed = 0;
    const auto consume = [&](JobResult&& r) {
        const auto [is_graphics, index] = job_kinds[consumed++];
        if (!r.ok) {
            ++result.skipped;
            return;
        }
        if (!is_graphics) {
            const std::string key(reinterpret_cast<const char*>(r.compute_key.data()),
                                  r.compute_key.size());
            if (compute_keys.insert(key).second) {
                compute_out.write(reinterpret_cast<const char*>(r.bytes.data()),
                                  static_cast<std::streamsize>(r.bytes.size()));
                ++result.compute_programs;
            }
            return;
        }
        GraphicsProgram& program = graphics[index];
        program.built = true;
        program.hash = r.hash;
        if (program.stage == 4) {
            if (!fragment_hashes.insert(r.hash).second) {
                return; // same key as an earlier entry
            }
            const auto key = GraphicsEntryKey(4, fragment_template, r.hash, 0);
            Write(graphics_out, u32{1});
            graphics_out.write(reinterpret_cast<const char*>(r.bytes.data()),
                               static_cast<std::streamsize>(r.bytes.size()));
            graphics_out.write(reinterpret_cast<const char*>(key.data()),
                               static_cast<std::streamsize>(key.size()));
            ++result.fragment_programs;
            return;
        }
        if (!vertex_hashes.insert(r.hash).second) {
            // Same stage hash: same key only if the attribute guess matches, known at the end.
            held_vertices.push_back({index, std::move(r.bytes)});
            return;
        }
        const auto key = GraphicsEntryKey(0, vertex_template, r.hash, 0);
        Write(graphics_out, u32{1});
        graphics_out.write(reinterpret_cast<const char*>(r.bytes.data()),
                           static_cast<std::streamsize>(r.bytes.size()));
        const auto key_offset = static_cast<u64>(graphics_out.tellp());
        graphics_out.write(reinterpret_cast<const char*>(key.data()),
                           static_cast<std::streamsize>(key.size()));
        written_vertices.push_back({index, key_offset});
    };
    const auto drain = [&](bool wait_all) {
        while (auto r = converter->Next(wait_all)) {
            consume(std::move(*r));
        }
    };

    for (size_t archive_index = 0; archive_index < archives.size(); ++archive_index) {
        if (stop.stop_requested()) {
            converter.reset();
            return fail("Stopped");
        }
        report(StageExtracting, archive_index, archives.size());
        const auto& archive = archives[archive_index];
        const auto data = archive.file->ReadAllBytes();
        const auto name = archive.path.substr(archive.path.rfind('/') + 1);
        const auto on_record = [&](const ShaderRecord& record) {
            const bool is_graphics =
                record.stage == ProgramStage::Vertex || record.stage == ProgramStage::Fragment;
            if (record.stage != ProgramStage::Compute && !is_graphics) {
                return;
            }
            const u128 code = Hash128(record.code);
            const u128 control = Hash128(record.control);
            Job job;
            if (is_graphics) {
                const auto [it, inserted] = graphics_index.try_emplace(code, graphics.size());
                if (!inserted) {
                    GraphicsProgram& program = graphics[it->second];
                    if (program.control == control) {
                        program.archives.push_back(static_cast<u32>(archive_index));
                    }
                    return;
                }
                graphics.push_back({record.stage == ProgramStage::Vertex ? 0U : 4U, control,
                                    {static_cast<u32>(archive_index)}});
                job.graphics = true;
                job.stage = graphics.back().stage;
                job.graphics_index = it->second;
            } else {
                if (!compute_identities.insert(PairHash{code, control}).second) {
                    return;
                }
                const auto values = ReadComputeControl(record.control);
                if (!values || !values->workgroup[0] || !values->workgroup[1] ||
                    !values->workgroup[2]) {
                    ++result.skipped;
                    return;
                }
                job.control = *values;
                job.stage = 5;
            }
            job.seq = seq++;
            job.program.assign(record.code.begin(), record.code.end());
            job_kinds.emplace_back(job.graphics, job.graphics_index);
            converter->Push(std::move(job));
            while (converter->Pending() > MaxPendingJobs && !stop.stop_requested()) {
                if (auto r = converter->Next(true)) {
                    consume(std::move(*r));
                }
            }
        };
        const auto on_member_error = [&](const std::string& error) {
            ++member_errors;
            LOG_DEBUG(Render_Vulkan, "Shader warm-up: {}: {}", archive.path, error);
        };
        try {
            WalkArchive(data, name, &dictionaries, on_record, on_member_error);
        } catch (const ArchiveError& e) {
            ++archive_errors;
            LOG_WARNING(Render_Vulkan, "Shader warm-up: {}: {}", archive.path, e.what());
        }
        drain(false);
    }
    report(StageExtracting, archives.size(), archives.size());

    const size_t total_jobs = seq;
    while (consumed < total_jobs) {
        if (stop.stop_requested()) {
            converter.reset();
            return fail("Stopped");
        }
        report(StageConverting, consumed, total_jobs);
        if (auto r = converter->Next(true)) {
            consume(std::move(*r));
        }
    }
    report(StageConverting, total_jobs, total_jobs);
    converter.reset();
    for (auto& archive : archives) {
        archive.file = nullptr;
    }
    effects.clear();
    dictionary_pack = nullptr;
    game.romfs.clear();
    game.loader.reset();
    game.nup.reset();

    // Vertex attribute types (make_graphics_warm.py attr_types): per slot, the most common
    // nonzero value among the runtime pipelines of the OTHER vertex shaders of the same
    // archive(s); slots without data are 1 (float). Without runtime data: built-in default.
    std::unordered_map<std::string, u32> archive_ids;
    for (size_t i = 0; i < archives.size(); ++i) {
        archive_ids.try_emplace(archives[i].path, static_cast<u32>(archive_ids.size()));
    }
    const auto archive_id = [&](u32 index) { return archive_ids.at(archives[index].path); };
    // Archive identity: the path inside the RomFS (the same file in both packages is one).
    std::unordered_map<u32, std::set<u64>> by_archive;
    for (const auto& program : graphics) {
        if (program.stage == 0 && program.built &&
            stats.vertex_attributes.contains(program.hash)) {
            for (const u32 a : program.archives) {
                by_archive[archive_id(a)].insert(program.hash);
            }
        }
    }
    const auto attribute_types = [&](const GraphicsProgram& program) -> u64 {
        if (!use_stats) {
            return DefaultAttributeTypes;
        }
        std::set<u64> others;
        for (const u32 a : program.archives) {
            if (const auto it = by_archive.find(archive_id(a)); it != by_archive.end()) {
                others.insert(it->second.begin(), it->second.end());
            }
        }
        others.erase(program.hash);
        std::array<std::array<size_t, 4>, 32> counts{};
        std::array<std::array<size_t, 4>, 32> first{};
        size_t order = 0;
        for (const u64 hash : others) {
            for (const u64 value : stats.vertex_attributes.at(hash)) {
                for (size_t i = 0; i < 32; ++i) {
                    const auto x = static_cast<size_t>((value >> (2 * i)) & 3);
                    if (x != 0) {
                        if (counts[i][x]++ == 0) {
                            first[i][x] = order++;
                        }
                    }
                }
            }
        }
        u64 types = 0;
        for (size_t i = 0; i < 32; ++i) {
            u64 best = 1;
            size_t best_count = 0;
            size_t best_first = 0;
            for (size_t x = 1; x < 4; ++x) {
                if (counts[i][x] > best_count ||
                    (counts[i][x] == best_count && counts[i][x] != 0 && first[i][x] < best_first)) {
                    best = x;
                    best_count = counts[i][x];
                    best_first = first[i][x];
                }
            }
            types |= best << (2 * i);
        }
        return types;
    };

    report(StageWriting, 0, written_vertices.size() + held_vertices.size());
    std::set<std::pair<u64, u64>> vertex_keys; // (stage hash, attribute types)
    std::vector<std::pair<u64, u64>> patches;  // key offset, attribute types
    for (const auto& written : written_vertices) {
        const auto& program = graphics[written.program];
        const u64 types = attribute_types(program);
        vertex_keys.emplace(program.hash, types);
        patches.emplace_back(written.key_offset + AttributeTypesOffset, types);
    }
    for (auto& held : held_vertices) {
        const auto& program = graphics[held.program];
        const u64 types = attribute_types(program);
        if (!vertex_keys.emplace(program.hash, types).second) {
            continue;
        }
        const auto key = GraphicsEntryKey(0, vertex_template, program.hash, types);
        Write(graphics_out, u32{1});
        graphics_out.write(reinterpret_cast<const char*>(held.env.data()),
                           static_cast<std::streamsize>(held.env.size()));
        graphics_out.write(reinterpret_cast<const char*>(key.data()),
                           static_cast<std::streamsize>(key.size()));
        held.env = {};
    }
    result.vertex_programs = vertex_keys.size();
    compute_out.close();
    graphics_out.close();
    if (!compute_out || !graphics_out) {
        return fail("Writing the warm-up files failed");
    }
    {
        std::fstream patch(graphics_tmp, std::ios::binary | std::ios::in | std::ios::out);
        if (!patch) {
            return fail("Cannot reopen " + Common::FS::PathToUTF8String(graphics_tmp));
        }
        size_t done = 0;
        for (const auto& [offset, types] : patches) {
            patch.seekp(static_cast<std::streamoff>(offset));
            patch.write(reinterpret_cast<const char*>(&types), sizeof(types));
            if (++done % 4096 == 0) {
                report(StageWriting, done, patches.size());
            }
        }
        if (!patch) {
            return fail("Writing the warm-up files failed");
        }
    }
    // New files: the markers of the old ones no longer apply.
    fs::remove(WarmMarkerPath(compute_path), ec);
    fs::remove(WarmMarkerPath(graphics_path), ec);
    fs::rename(compute_tmp, compute_path, ec);
    if (ec) {
        return fail("Cannot replace " + Common::FS::PathToUTF8String(compute_path));
    }
    fs::rename(graphics_tmp, graphics_path, ec);
    if (ec) {
        return fail("Cannot replace " + Common::FS::PathToUTF8String(graphics_path));
    }
    report(StageWriting, patches.size(), patches.size());
    result.ok = true;
    LOG_INFO(Render_Vulkan,
             "Shader warm-up build for {:016x}: {} compute, {} vertex, {} fragment entries, {} "
             "programs skipped, {} archive and {} member errors, {:.1f} s",
             title_id, result.compute_programs, result.vertex_programs, result.fragment_programs,
             result.skipped, archive_errors, member_errors,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return result;
}

} // namespace VideoCommon::ShaderWarm
