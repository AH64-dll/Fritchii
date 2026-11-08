// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <utility>

#include "common/bit_field.h"
#include "common/common_types.h"
#include "common/scratch_buffer.h"

namespace Tegra {
class MemoryManager;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra::Engines::Upload {

struct Registers {
    u32 line_length_in;
    u32 line_count;

    struct {
        u32 address_high;
        u32 address_low;
        u32 pitch;
        union {
            BitField<0, 4, u32> block_width;
            BitField<4, 4, u32> block_height;
            BitField<8, 4, u32> block_depth;
        };
        u32 width;
        u32 height;
        u32 depth;
        u32 layer;
        u32 x;
        u32 y;

        GPUVAddr Address() const {
            return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
        }

        u32 BlockWidth() const {
            return block_width.Value();
        }

        u32 BlockHeight() const {
            return block_height.Value();
        }

        u32 BlockDepth() const {
            return block_depth.Value();
        }
    } dest;
};

/// The guest memory [first, second) inline data can land in with these registers, for a copy
/// launched with `is_linear` (pitch layout) and `copy_size` (line_length_in * line_count then):
/// the split GPU thread's front bounds the uploads it queues with it (GpuSplit::NoteWrite).
[[nodiscard]] std::pair<GPUVAddr, GPUVAddr> WriteRange(const Registers& regs, bool is_linear,
                                                       u32 copy_size);

class State {
public:
    explicit State(MemoryManager& memory_manager_, Registers& regs_);
    ~State();

    void SetNativePartialTransfers(bool enabled) { native_partial = enabled; }
    void ProcessExec(bool is_linear_);
    void ProcessData(u32 data, bool is_last_call);
    void ProcessData(const u32* data, size_t num_data);

    /// EDEN_NX2_COMPUTE_I2M_MERGE (KeplerCompute::BackWrites): while held, a partial linear
    /// transfer's flush that continues the held bytes within their 4 KiB page joins them instead
    /// of being written. The held bytes are written when holding ends and before any other
    /// write. The caller holds only while nothing else reads or writes memory.
    void HoldWrites(bool hold) {
        if (!hold) {
            FlushHeld();
        }
        holding = hold;
    }

    /// Binds a rasterizer to this engine.
    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    GPUVAddr ExecTargetAddress() const {
        return regs.dest.Address();
    }

    u32 GetUploadSize() const {
        return copy_size;
    }

private:
    void ProcessData(std::span<const u8> read_buffer);
    /// A partial linear transfer's flush (AccelerateInlineToMemory), joined while holding.
    void Emit(GPUVAddr address, std::span<const u8> bytes);
    void FlushHeld();

    static constexpr u32 HeldPage = 0x1000;
    bool holding{};
    u32 held_size{};
    GPUVAddr held_address{};
    std::array<u8, HeldPage> held_bytes;

    bool native_partial{};
    u32 flushed_offset{};
    u32 write_offset = 0;
    u32 copy_size = 0;
    Common::ScratchBuffer<u8> inner_buffer;
    Common::ScratchBuffer<u8> tmp_buffer;
    bool is_linear = false;
    Registers& regs;
    MemoryManager& memory_manager;
    VideoCore::RasterizerInterface* rasterizer = nullptr;
};

} // namespace Tegra::Engines::Upload
