// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "video_core/engines/maxwell_3d.h"

namespace Tegra {
class Memorymanager;
}

namespace VideoCommon {

/// A native (Switch 2) program after capture and lowering to Maxwell code. The shader cache keeps
/// it so the environments built for the program's pipelines don't capture and lower it again.
struct LoweredNativeCode {
    u64 native_program_key{};
    /// Shader header, lowered instructions with Maxwell scheduling words, then the native
    /// address map (what the environment's `code` used to be filled with).
    std::vector<u64> code;
    std::vector<Shader::Ampere::Instruction> trap_prefix;
    /// Bytes of the native program that were read, header included.
    size_t source_size{};
    /// Constant buffer holding the texture handles if the program requires one, otherwise ~0U
    /// (the engine's register value is used).
    u32 texture_bound = ~0U;
};

class GenericEnvironment : public Shader::Environment {
public:
    explicit GenericEnvironment() = default;
    explicit GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_);

    ~GenericEnvironment() override;

    [[nodiscard]] u32 TextureBoundBuffer() const final;

    [[nodiscard]] u32 LocalMemorySize() const final;

    [[nodiscard]] u32 SharedMemorySize() const final;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const final;

    [[nodiscard]] u64 ReadInstruction(u32 address) final;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;

    [[nodiscard]] std::vector<u64> MappedStoragePointerKeys() const override;

    /// Answers the classification of these keys (see MappedStoragePointerKeys) as storage buffer
    /// pointers, whatever the slots hold now; called before the program is translated
    void KeepStoragePointers(std::span<const u64> keys);

    [[nodiscard]] std::optional<u64> Analyze();

    void SetCachedSize(size_t size_bytes);

    [[nodiscard]] size_t CachedSizeWords() const noexcept;

    [[nodiscard]] size_t CachedSizeBytes() const noexcept;

    [[nodiscard]] size_t ReadSizeBytes() const noexcept;

    [[nodiscard]] size_t SourceSizeBytes() const noexcept {
        return native_source_size;
    }

    [[nodiscard]] const std::shared_ptr<const LoweredNativeCode>& LoweredNative() const noexcept {
        return lowered_native;
    }

    [[nodiscard]] bool CanBeSerialized() const noexcept;

    [[nodiscard]] u64 CalculateHash() const;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    void Serialize(std::ofstream& file) const;

    bool HasHLEMacroState() const override {
        return has_hle_engine_state;
    }
    bool HasNativeInstructions() const override;

    /// Whether a guarded guest memory read gave up (a page table writer kept its lock): the
    /// translation saw zeros there, so its result must not be used.
    [[nodiscard]] bool GuardedReadFailed() const noexcept {
        return guarded_read_failed;
    }

    /// The first of the recorded reads (constant buffer values, texture types and formats,
    /// storage pointer classes, replacements, the viewport transform state) whose answer differs
    /// from `other`'s, described; empty when they all match.
    [[nodiscard]] std::string FirstReadDifference(const GenericEnvironment& other) const;

protected:
    /// ReadBlockGuarded of the memory manager; zeros and GuardedReadFailed when it gave up.
    void ReadGuestGuarded(GPUVAddr address, void* data, size_t size);
    /// IsFullyMappedRangeGuarded; false and GuardedReadFailed when it gave up.
    bool IsMappedGuarded(GPUVAddr address, size_t size);

    std::optional<u64> TryFindSize();
    virtual std::optional<u32> ReadTrapGuardConstant(u32, u32) { return std::nullopt; }

    /// Adopts an already lowered native program, or captures and lowers it when there is none.
    void InitializeNativeCode(std::shared_ptr<const LoweredNativeCode> lowered,
                              u32 source_header_size = 128, bool geometry_stage = false);
    std::shared_ptr<const LoweredNativeCode> LowerNativeCode(u32 source_header_size,
                                                          bool geometry_stage);
    size_t native_source_size{};
    std::shared_ptr<const LoweredNativeCode> lowered_native;

    /// `live`: read with ReadBlock even when the reads are guarded (the checks against the live
    /// state, on the GPU thread).
    Tegra::Texture::TICEntry ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                             bool via_header_index, u32 raw, bool live = false);

    Tegra::MemoryManager* gpu_memory{};
    GPUVAddr program_base{};

    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types; ///< IsCbufStorageBufferPointer answers
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;

    u32 local_memory_size{};
    u32 texture_bound{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    u32 read_lowest = (std::numeric_limits<u32>::max)();
    u32 read_highest = 0;

    u32 cached_lowest = (std::numeric_limits<u32>::max)();
    u32 cached_highest = 0;
    u32 initial_offset = 0;

    u32 viewport_transform_state = 1;

    bool has_unbound_instructions = false;
    bool has_hle_engine_state = false;
    /// Guest memory is read through the guarded reads (a snapshot environment, which may be
    /// translated off the GPU thread).
    bool guarded_reads = false;
    bool guarded_read_failed = false;
};

class GraphicsEnvironment final : public GenericEnvironment {
public:
    explicit GraphicsEnvironment() = default;
    explicit GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                 Tegra::MemoryManager& gpu_memory_,
                                 Tegra::Engines::Maxwell3D::Regs::ShaderType program,
                                 GPUVAddr program_base_, u32 start_address_,
                                 std::shared_ptr<const LoweredNativeCode> lowered_native_ = {});

    ~GraphicsEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;

    /// EDEN_NX2_ASYNC_TRANSLATE: copies what the translation reads from the engine and from
    /// constant buffers (the stage's constant buffer table and the bytes of each enabled buffer,
    /// the texture pool registers, the viewport transform state, the HLE replace entries), so the
    /// program can be translated later, on another thread, with the answers it would get now.
    /// From then on the environment never reads the engine, and reads guest memory only for
    /// texture descriptors, storage pointer checks and unbound instructions, through the guarded
    /// reads. GPU thread, right after construction. Returns the bytes copied.
    size_t TakeSnapshot();

    /// Frees the copied constant buffer bytes once translated (the recorded reads and the
    /// texture pool registers stay).
    void ReleaseSnapshotBytes();

    [[nodiscard]] bool HasSnapshot() const noexcept {
        return snapshot != nullptr;
    }

    struct LiveCheck {
        size_t reads{};     ///< Recorded reads answered again
        size_t differing{}; ///< Of them, answered differently now
        std::string first;  ///< The first that differs, described
    };
    /// GPU thread, right after translating from the snapshot: every recorded read answered again
    /// from the engine and guest memory as they are now (EDEN_NX2_ASYNC_TRANSLATE=3).
    [[nodiscard]] LiveCheck CheckAgainstLive();

    /// GPU thread, before a pipeline translated from the snapshot is used: the texture
    /// descriptors it read (types and formats), read again from the snapshot's pool with
    /// ReadBlock, still give what it recorded. False means the pool changed meanwhile (or a
    /// descriptor needed the cache flush the worker can't do).
    [[nodiscard]] bool TextureReadsStillMatch();

private:
    /// What TakeSnapshot copies.
    struct Snapshot {
        std::array<Tegra::Engines::ConstBufferInfo,
                   Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>
            const_buffers{};
        /// Bytes of each enabled buffer as Read<u32> sees them, one word past its size.
        std::array<std::vector<u32>, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers> words;
        ::Common::unordered_map<u64, Shader::ReplaceConstant> replace_entries;
        GPUVAddr tic_address{};
        u32 tic_limit{};
        bool via_header_index{};
        u32 viewport_transform_state{};
    };

    /// The engine's answers as they are now, without recording them.
    [[nodiscard]] u32 LiveCbufValue(u32 cbuf_index, u32 cbuf_offset) const;
    [[nodiscard]] bool LiveStoragePointer(u64 key) const;
    [[nodiscard]] std::optional<Shader::ReplaceConstant> LiveReplaceConstBuffer(u32 bank,
                                                                                u32 offset) const;
    [[nodiscard]] Tegra::Texture::TICEntry LiveTextureInfo(u32 handle);

    Tegra::Engines::Maxwell3D* maxwell3d{};
    size_t stage_index{};
    /// Shared: the environments are copied (they declare a destructor, so they have no moves).
    std::shared_ptr<Snapshot> snapshot;
};

class ComputeEnvironment final : public GenericEnvironment {
public:
    explicit ComputeEnvironment() = default;
    explicit ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                u32 start_address_,
                                std::shared_ptr<const LoweredNativeCode> lowered_native_ = {});

    ~ComputeEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    Shader::TextureType ReadTextureType(u32 handle) override;

    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    bool IsTexturePixelFormatInteger(u32 handle) override;

    u32 ReadViewportTransformState() override;

    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(
        [[maybe_unused]] u32 bank, [[maybe_unused]] u32 offset) override {
        return std::nullopt;
    }

private:
    std::optional<u32> ReadTrapGuardConstant(u32 bank, u32 offset) override;
    Tegra::Engines::KeplerCompute* kepler_compute{};
};

std::optional<u32> ReadNativeGuardConstant(Tegra::Engines::KeplerCompute& compute,
                                         Tegra::MemoryManager& memory, u32 bank, u32 offset);

/// Answers for reads a FileEnvironment has no record of (EDEN_NX2_COMPUTE_WARM: programs from the
/// game files, never run, converted only to fill the driver's cache). Without it a missing read
/// is an error.
struct GuessMissing {
    u32 cbuf_bank0{0x10000000};  ///< Constant buffer 0 (driver constants, descriptor handles)
    u32 cbuf_other{0};           ///< Other constant buffers (game constants)
    Shader::TextureType texture_type{Shader::TextureType::ColorArray2D};
    Shader::TexturePixelFormat pixel_format{static_cast<Shader::TexturePixelFormat>(16)};
    bool storage_pointer_bank0{true};  ///< Bank 0 always held storage pointers in the test title's records
    bool storage_pointer_other{false}; ///< Banks 1-4 never did (95% identical SPIR-V offline)
};

class FileEnvironment final : public Shader::Environment {
public:
    FileEnvironment() = default;

    /// Missing reads get these answers instead of an error; the guess must outlive the
    /// environment.
    void SetGuessMissing(const GuessMissing* guess_) noexcept {
        guess = guess_;
    }
    ~FileEnvironment() override = default;

    FileEnvironment& operator=(FileEnvironment&&) noexcept = default;
    FileEnvironment(FileEnvironment&&) noexcept = default;

    FileEnvironment& operator=(const FileEnvironment&) = delete;
    FileEnvironment(const FileEnvironment&) = delete;

    void Deserialize(std::istream& file);

    /// Whether the record at the file's read position (what Deserialize reads) can fit in the
    /// `bytes_left` bytes before the end of the file: its code size and table counts are peeked
    /// and the read position is left where it was. A corrupt record's counts would otherwise
    /// make Deserialize allocate gigabytes.
    [[nodiscard]] static bool RecordFits(std::istream& file, u64 bytes_left);

    [[nodiscard]] u64 ReadInstruction(u32 address) override;

    [[nodiscard]] u32 TranslateProgramAddress(u32 address) const override;

    [[nodiscard]] bool IsNativeProgram() const override;

    [[nodiscard]] bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;

    [[nodiscard]] bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset,
                                                          u32 indirect) override;

    [[nodiscard]] std::vector<std::pair<u32, u32>> NullStoragePointerSlots() const override;

    [[nodiscard]] std::vector<u64> MappedStoragePointerKeys() const override;

    [[nodiscard]] u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;

    [[nodiscard]] Shader::TextureType ReadTextureType(u32 handle) override;

    [[nodiscard]] Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;

    [[nodiscard]] bool IsTexturePixelFormatInteger(u32 handle) override;

    [[nodiscard]] u32 ReadViewportTransformState() override;

    [[nodiscard]] u32 LocalMemorySize() const override;

    [[nodiscard]] u32 SharedMemorySize() const override;

    [[nodiscard]] u32 TextureBoundBuffer() const override;

    [[nodiscard]] std::array<u32, 3> WorkgroupSize() const override;

    [[nodiscard]] std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) override;

    [[nodiscard]] bool HasHLEMacroState() const override {
        return cbuf_replacements.size() != 0;
    }
    bool HasNativeInstructions() const override;

    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    /// Whether translating this recorded environment gives what translating `live`, the same
    /// program as the game draws it now, would (zortos293/Lichii#896; the same code is assumed:
    /// the pipeline keys match). Translation is decided by the answers to its questions, so each
    /// recorded question is asked of `live`: a constant buffer value must be the same, unless it
    /// was only used as a texture handle (then the texture it names now must have the same type
    /// and the same pixel format traits as the recorded one) or as a storage buffer pointer word
    /// (then its classification must be the same); texture questions, storage pointer
    /// classifications and the other recorded answers must be the same. Asking them also leaves
    /// `live` holding every answer a translation of it would have recorded (what the disk cache
    /// stores). Sets `why` to the first answer that differs. `format_traits` limits the pixel
    /// format traits compared to those the program's translation used (see the FormatTrait bits).
    [[nodiscard]] bool AnswersHoldFor(Shader::Environment& live, const char** why = nullptr,
                                      u32 format_traits = ~0U);

    /// Pixel format traits translation uses: integer formats (sampled textures).
    static constexpr u32 FormatTraitInteger = 1;
    /// R16_FLOAT and the byte-addressed 32-bit color formats (byte-addressed image stores).
    static constexpr u32 FormatTraitByteAddressed = 2 | 4;
    /// Snorm formats (buffer texel fetches on hosts without snorm render buffers).
    static constexpr u32 FormatTraitSnorm = 8;

private:
    std::vector<u64> code;
    ::Common::unordered_map<u32, Shader::TextureType> texture_types;
    ::Common::unordered_map<u32, Shader::TexturePixelFormat> texture_pixel_formats;
    ::Common::unordered_map<u64, u32> cbuf_values;
    ::Common::unordered_map<u64, u8> storage_pointer_types;
    ::Common::unordered_map<u64, Shader::ReplaceConstant> cbuf_replacements;
    std::array<u32, 3> workgroup_size{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    u32 texture_bound{};
    u32 read_lowest{};
    u32 read_highest{};
    u32 initial_offset{};
    u32 viewport_transform_state = 1;
    const GuessMissing* guess{};
};

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version);

template <typename Key, typename Envs>
    requires std::is_trivially_copyable_v<Key>
        && std::has_unique_object_representations_v<Key>
void SerializePipeline(const Key& key, const Envs& envs, const std::filesystem::path& filename, u32 cache_version) {
    SerializePipeline(std::span(reinterpret_cast<const char*>(&key), sizeof(key)), std::span(envs.data(), envs.size()), filename, cache_version);
}

void LoadPipelines(std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics);

} // namespace VideoCommon
