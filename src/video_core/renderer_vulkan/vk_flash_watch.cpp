// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>

#include <boost/container/small_vector.hpp>
#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host_shaders/vulkan_flash_watch_comp_spv.h"
#include "video_core/renderer_vulkan/vk_flash_watch.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

using VideoCore::Surface::PixelFormat;
using Kind = FlashWatch::Kind;
using Output = FlashWatch::Output;
using Tile = FlashWatch::Tile;

constexpr u32 GRID_X = 32;
constexpr u32 GRID_Y = 18;
/// Frames in flight before a readback waits for the GPU.
constexpr u32 SLOTS = 4;
constexpr u32 SLOT_TILES = 1U << 18;
constexpr u32 MAX_PASS_SHADERS = 32;
/// Frames N-2..N+2 decide whether frame N flashed.
constexpr size_t WINDOW = 5;
constexpr size_t CENTER = 2;
constexpr u64 STATUS_FRAMES = 3600;
constexpr u64 ALTERNATION_FRAMES = 300;

struct PushConstants {
    std::array<u32, 2> size;
    std::array<u32, 2> grid;
    u32 base;
    u32 channels;
    f32 hot;
    u32 padding;
};

/// Channels that take part in the mean and the largest magnitude; alpha never does.
u32 Channels(PixelFormat format) {
    switch (format) {
    case PixelFormat::R8_UNORM:
    case PixelFormat::R8_SNORM:
    case PixelFormat::R16_FLOAT:
    case PixelFormat::R16_UNORM:
    case PixelFormat::R16_SNORM:
    case PixelFormat::R32_FLOAT:
        return 1;
    case PixelFormat::R8G8_UNORM:
    case PixelFormat::R8G8_SNORM:
    case PixelFormat::R16G16_FLOAT:
    case PixelFormat::R16G16_UNORM:
    case PixelFormat::R16G16_SNORM:
    case PixelFormat::R32G32_FLOAT:
    case PixelFormat::G4R4_UNORM:
        return 2;
    default:
        return 3;
    }
}

double EnvDouble(const char* name, double fallback) {
    const char* const value = std::getenv(name);
    if (!value || *value == '\0') {
        return fallback;
    }
    char* end{};
    const double parsed = std::strtod(value, &end);
    return end != value ? parsed : fallback;
}

std::optional<FlashWatch::Source> MakeSource(ImageView& view, u32 slot) {
    using namespace VideoCore::Surface;
    const Image* const image = view.BackingImage();
    if (!image) {
        return std::nullopt;
    }
    const auto& info = image->info;
    if (info.type == VideoCommon::ImageType::Buffer || info.num_samples > 1 ||
        GetFormatType(view.format) != SurfaceType::ColorTexture ||
        IsPixelFormatInteger(view.format) || DefaultBlockWidth(view.format) != 1) {
        return std::nullopt;
    }
    const VkImageView handle = view.Handle(Shader::TextureType::Color2D);
    if (handle == VK_NULL_HANDLE) {
        return std::nullopt;
    }
    const bool rescaled = image->IsRescaled();
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = rescaled ? resolution.ScaleUp(info.size.width) : info.size.width;
    const u32 height = rescaled ? resolution.ScaleUp(info.size.height) : info.size.height;
    const u32 level = static_cast<u32>(view.range.base.level);
    return FlashWatch::Source{
        .view = handle,
        .format = view.format,
        .width = std::max(1U, width >> level),
        .height = std::max(1U, height >> level),
        .gpu_addr = image->gpu_addr,
        .slot = slot,
    };
}

struct Stats {
    double sum{};
    u64 texels{}; ///< Finite texels
    double peak{};
    u64 bad{};
    u64 hot{};

    [[nodiscard]] double Mean() const {
        return texels != 0 ? sum / static_cast<double>(texels) : 0.0;
    }
};

/// Normalised [x0, x1) x [y0, y1).
struct Region {
    double x0;
    double y0;
    double x1;
    double y1;
};
constexpr Region WHOLE{0.0, 0.0, 1.0, 1.0};

/// Same split as the shader.
u64 TileTexels(const Output& output, u32 x, u32 y) {
    const u64 x0 = u64{x} * output.width / output.grid_x;
    const u64 x1 = u64{x + 1} * output.width / output.grid_x;
    const u64 y0 = u64{y} * output.height / output.grid_y;
    const u64 y1 = u64{y + 1} * output.height / output.grid_y;
    return (x1 - x0) * (y1 - y0);
}

void Add(Stats& stats, const Output& output, std::span<const Tile> tiles, u32 x, u32 y) {
    const Tile& tile = tiles[output.first_tile + y * output.grid_x + x];
    const u64 texels = TileTexels(output, x, y);
    stats.sum += std::bit_cast<f32>(tile[0]);
    stats.peak = std::max<double>(stats.peak, std::bit_cast<f32>(tile[1]));
    stats.bad += tile[2];
    stats.hot += tile[3];
    stats.texels += texels > tile[2] ? texels - tile[2] : 0;
}

Stats Gather(const Output& output, std::span<const Tile> tiles, const Region& region) {
    Stats stats;
    for (u32 y = 0; y < output.grid_y; ++y) {
        const double y0 = static_cast<double>(y) / output.grid_y;
        const double y1 = static_cast<double>(y + 1) / output.grid_y;
        if (y1 <= region.y0 || y0 >= region.y1) {
            continue;
        }
        for (u32 x = 0; x < output.grid_x; ++x) {
            const double x0 = static_cast<double>(x) / output.grid_x;
            const double x1 = static_cast<double>(x + 1) / output.grid_x;
            if (x1 > region.x0 && x0 < region.x1) {
                Add(stats, output, tiles, x, y);
            }
        }
    }
    return stats;
}

using Window = std::array<std::optional<double>, WINDOW>;

/// The larger of the brightest frame before the centre and the darkest after it: a flash stands
/// above both, while a cut or a fade keeps its new level in one of the frames after.
std::optional<double> Reference(const Window& values) {
    std::optional<double> before;
    std::optional<double> after;
    for (size_t f = 0; f < CENTER; ++f) {
        if (values[f]) {
            before = std::max(before.value_or(*values[f]), *values[f]);
        }
    }
    for (size_t f = CENTER + 1; f < WINDOW; ++f) {
        if (values[f]) {
            after = std::min(after.value_or(*values[f]), *values[f]);
        }
    }
    if (!before || !after) {
        return std::nullopt;
    }
    return std::max(*before, *after);
}

/// A count of NaN or hot texels that appeared or at least doubled.
bool Spiked(double now, double reference) {
    return now > reference && now >= 2.0 * reference;
}

std::string_view KindName(Kind kind) {
    switch (kind) {
    case Kind::Pass:
        return "pass";
    case Kind::Dispatch:
        return "dispatch";
    case Kind::Present:
        return "present";
    }
    return "unknown";
}

std::string ShaderList(std::span<const u64> shaders, size_t limit, bool quoted) {
    std::string out;
    for (size_t i = 0; i < shaders.size() && i < limit; ++i) {
        out += i == 0 ? "" : ",";
        out += quoted ? fmt::format("\"{:016x}\"", shaders[i]) : fmt::format("{:016x}", shaders[i]);
    }
    if (!quoted && shaders.size() > limit) {
        out += fmt::format(",+{}", shaders.size() - limit);
    }
    return out;
}

std::string Describe(const Output& output) {
    switch (output.kind) {
    case Kind::Pass:
        return fmt::format("pass {} rt{} {} {}x{} at {:#x} ({} draws, shaders {})", output.index,
                           output.slot, output.format, output.width, output.height,
                           output.gpu_addr, output.draws, ShaderList(output.shaders, 6, false));
    case Kind::Dispatch:
        return fmt::format("dispatch {} {} img{} {} {}x{} at {:#x}", output.index,
                           ShaderList(output.shaders, 1, false), output.slot, output.format,
                           output.width, output.height, output.gpu_addr);
    case Kind::Present:
        return fmt::format("present {} {} {}x{} at {:#x}", output.index, output.format,
                           output.width, output.height, output.gpu_addr);
    }
    return {};
}

std::string JsonNumber(double value) {
    return std::isfinite(value) ? fmt::format("{:.6g}", value) : "null";
}

std::string StatsJson(const std::array<std::optional<Stats>, WINDOW>& stats) {
    const auto list = [&stats](auto&& get) {
        std::string out{"["};
        for (size_t f = 0; f < WINDOW; ++f) {
            out += f == 0 ? "" : ",";
            out += stats[f] ? get(*stats[f]) : "null";
        }
        return out + "]";
    };
    return fmt::format(
        R"({{"mean":{},"peak":{},"bad":{},"hot":{}}})",
        list([](const Stats& s) { return JsonNumber(s.Mean()); }),
        list([](const Stats& s) { return JsonNumber(s.peak); }),
        list([](const Stats& s) { return fmt::format("{}", s.bad); }),
        list([](const Stats& s) { return fmt::format("{}", s.hot); }));
}

} // Anonymous namespace

FlashWatch::FlashWatch(const Device& device_, MemoryAllocator& memory_allocator_,
                       Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      descriptor_pool{device_, scheduler_} {
    const char* const dir = std::getenv("EDEN_NX2_FLASH_WATCH");
    if (!dir || *dir == '\0') {
        return;
    }
    directory = dir;
    const char* const passes_value = std::getenv("EDEN_NX2_FLASH_WATCH_PASSES");
    passes = !passes_value || std::string_view{passes_value} != "0";
    hot = static_cast<float>(EnvDouble("EDEN_NX2_FLASH_WATCH_HOT", 1024.0));
    ratio = EnvDouble("EDEN_NX2_FLASH_WATCH_RATIO", 1.5);
    delta = EnvDouble("EDEN_NX2_FLASH_WATCH_DELTA", 0.15);
    pass_ratio = EnvDouble("EDEN_NX2_FLASH_WATCH_PASS_RATIO", 2.0);
    max_reports = static_cast<u32>(EnvDouble("EDEN_NX2_FLASH_WATCH_MAX", 10.0));
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    try {
        const vk::Device& dev = device.GetLogical();
        const std::array bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
        };
        set_layout = dev.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        layout = dev.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = set_layout.address(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        });
        module = dev.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = sizeof(VULKAN_FLASH_WATCH_COMP_SPV),
            .pCode = VULKAN_FLASH_WATCH_COMP_SPV,
        });
        pipeline = dev.CreateComputePipeline(
            {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *module,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
                .layout = *layout,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0,
            },
            VK_NULL_HANDLE);
        sampler = dev.CreateSampler({
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_NEVER,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        });
        descriptor_allocator = descriptor_pool.Allocator(device, scheduler, *set_layout,
                                                         DescriptorBankInfo{
                                                             .storage_buffers = 1,
                                                             .textures = 1,
                                                             .score = 2,
                                                         });
        slots.resize(SLOTS);
        for (Slot& slot : slots) {
            slot.buffer = memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = VkDeviceSize{SLOT_TILES} * sizeof(Tile),
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                },
                MemoryUsage::Download);
        }
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "Flash watch: setup failed ({}), watching nothing",
                  exception.what());
        return;
    }
    enabled = true;
    start = std::chrono::steady_clock::now();
    worker = std::thread{[this] { WorkerLoop(); }};
    LOG_INFO(Render_Vulkan,
             "Flash watch armed: dir={} passes={} ratio={} delta={} pass_ratio={} hot={} "
             "max_reports={}",
             directory.string(), passes, ratio, delta, pass_ratio, hot, max_reports);
}

FlashWatch::~FlashWatch() {
    if (!worker.joinable()) {
        return;
    }
    {
        std::scoped_lock lock{queue_mutex};
        stop_worker = true;
    }
    queue_ready.notify_one();
    worker.join();
}

bool FlashWatch::BeginPass(const void* framebuffer, std::span<ImageView* const> colors) {
    if (framebuffer == pass_framebuffer) {
        return false;
    }
    const bool recorded = EndPass();
    pass_framebuffer = framebuffer;
    pass_sources.clear();
    for (u32 slot = 0; slot < colors.size(); ++slot) {
        if (!colors[slot]) {
            continue;
        }
        if (const auto source = MakeSource(*colors[slot], slot)) {
            pass_sources.push_back(*source);
        }
    }
    pass_hashes.clear();
    pass_first_draw = draws;
    return recorded;
}

void FlashWatch::NoteDraw(std::span<const u64> shader_hashes) {
    ++draws;
    for (const u64 hash : shader_hashes) {
        if (hash != 0 && pass_hashes.size() < MAX_PASS_SHADERS &&
            std::ranges::find(pass_hashes, hash) == pass_hashes.end()) {
            pass_hashes.push_back(hash);
        }
    }
}

bool FlashWatch::EndPass() {
    if (!pass_framebuffer) {
        return false;
    }
    pass_framebuffer = nullptr;
    const u32 pass = pass_index++;
    // Passes without draws only clear.
    if (draws == pass_first_draw) {
        return false;
    }
    return Reduce(Kind::Pass, pass, pass_sources, pass_hashes, draws - pass_first_draw);
}

void FlashWatch::NoteDispatch(u64 shader_hash, std::span<ImageView* const> written) {
    const u32 dispatch = dispatches++;
    boost::container::small_vector<Source, 8> sources;
    for (u32 slot = 0; slot < written.size(); ++slot) {
        if (!written[slot]) {
            continue;
        }
        if (const auto source = MakeSource(*written[slot], slot)) {
            sources.push_back(*source);
        }
    }
    Reduce(Kind::Dispatch, dispatch, sources, std::array{shader_hash}, 0);
}

void FlashWatch::NotePresent(ImageView& view) {
    EndPass();
    const u32 present = presents++;
    if (const auto source = MakeSource(view, 0)) {
        Reduce(Kind::Present, present, std::span{&*source, 1}, {}, 0);
    }
}

bool FlashWatch::Reduce(Kind kind, u32 index, std::span<const Source> sources,
                        std::span<const u64> shaders, u32 draw_count) {
    Slot& slot = slots[current];
    boost::container::small_vector<std::pair<VkImageView, PushConstants>, 8> work;
    for (const Source& source : sources) {
        const u32 grid_x = std::min(GRID_X, source.width);
        const u32 grid_y = std::min(GRID_Y, source.height);
        if (slot.tiles + grid_x * grid_y > SLOT_TILES) {
            if (!std::exchange(warned_full, true)) {
                LOG_WARNING(Render_Vulkan,
                            "Flash watch: a frame has more than {} tiles; later outputs are "
                            "not watched",
                            SLOT_TILES);
            }
            continue;
        }
        std::string key;
        switch (kind) {
        case Kind::Pass:
            key = fmt::format("pass:{}:{}x{}:rt{}", source.format, source.width, source.height,
                              source.slot);
            break;
        case Kind::Dispatch:
            key = fmt::format("dispatch:{:016x}:{}:{}x{}:img{}", shaders.empty() ? 0 : shaders[0],
                              source.format, source.width, source.height, source.slot);
            break;
        case Kind::Present:
            key = fmt::format("present:{}:{}x{}", source.format, source.width, source.height);
            break;
        }
        key += fmt::format("#{}", key_counts[key]++);
        slot.outputs.push_back(Output{
            .kind = kind,
            .index = index,
            .slot = source.slot,
            .draws = draw_count,
            .format = source.format,
            .width = source.width,
            .height = source.height,
            .grid_x = grid_x,
            .grid_y = grid_y,
            .first_tile = slot.tiles,
            .gpu_addr = source.gpu_addr,
            .shaders = {shaders.begin(), shaders.end()},
            .key = std::move(key),
        });
        work.emplace_back(source.view, PushConstants{
                                           .size = {source.width, source.height},
                                           .grid = {grid_x, grid_y},
                                           .base = slot.tiles,
                                           .channels = Channels(source.format),
                                           .hot = hot,
                                           .padding = 0,
                                       });
        slot.tiles += grid_x * grid_y;
    }
    if (work.empty()) {
        return false;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, work = std::move(work), buffer = *slot.buffer](
                         vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, before);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        for (const auto& [view, push] : work) {
            const VkDescriptorSet set = descriptor_allocator.Commit();
            const VkDescriptorImageInfo image_info{
                .sampler = *sampler,
                .imageView = view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const VkDescriptorBufferInfo buffer_info{
                .buffer = buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            };
            const std::array writes{
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = set,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = set,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer_info,
                    .pTexelBufferView = nullptr,
                },
            };
            device.GetLogical().UpdateDescriptorSets(writes, {});
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, push);
            cmdbuf.Dispatch(push.grid[0], push.grid[1], 1);
        }
        // The game's next writes wait for these reads; the tiles become visible to the host.
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                               after);
    });
    return true;
}

void FlashWatch::TickFrame() {
    if (!enabled) {
        return;
    }
    EndPass();
    Slot& finished = slots[current];
    finished.tick = scheduler.CurrentTick();
    finished.frame = frame++;
    finished.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    finished.pending = true;
    current = (current + 1) % SLOTS;
    // Oldest first; the slot the next frame records into must be read now.
    for (u32 age = 0; age < SLOTS; ++age) {
        Slot& slot = slots[(current + age) % SLOTS];
        if (!slot.pending) {
            continue;
        }
        if (!scheduler.IsFree(slot.tick)) {
            if (age != 0) {
                break;
            }
            scheduler.Wait(slot.tick);
        }
        Collect(slot);
    }
    pass_index = 0;
    pass_first_draw = 0;
    draws = 0;
    dispatches = 0;
    presents = 0;
    key_counts.clear();
}

void FlashWatch::Collect(Slot& slot) {
    slot.buffer.Invalidate();
    Frame out{
        .frame = slot.frame,
        .seconds = slot.seconds,
        .outputs = std::move(slot.outputs),
        .tiles = std::vector<Tile>(slot.tiles),
        .keys = {},
    };
    std::memcpy(out.tiles.data(), slot.buffer.Mapped().data(), out.tiles.size() * sizeof(Tile));
    slot.outputs.clear();
    slot.tiles = 0;
    slot.pending = false;
    {
        std::scoped_lock lock{queue_mutex};
        queue.push_back(std::move(out));
    }
    queue_ready.notify_one();
}

void FlashWatch::WorkerLoop() {
    for (;;) {
        Frame next;
        {
            std::unique_lock lock{queue_mutex};
            queue_ready.wait(lock, [this] { return !queue.empty() || stop_worker; });
            if (queue.empty()) {
                return;
            }
            next = std::move(queue.front());
            queue.pop_front();
        }
        next.keys.reserve(next.outputs.size());
        for (u32 i = 0; i < next.outputs.size(); ++i) {
            next.keys.emplace(next.outputs[i].key, i);
        }
        if (next.frame % STATUS_FRAMES == 0 && !next.outputs.empty()) {
            LOG_INFO(Render_Vulkan,
                     "Flash watch: frame {} has {} outputs in {} tiles; {} frames checked, {} "
                     "flashes",
                     next.frame, next.outputs.size(), next.tiles.size(), checked, flashes);
        }
        history.push_back(std::move(next));
        if (history.size() > WINDOW) {
            history.pop_front();
        }
        if (history.size() == WINDOW &&
            history.back().frame - history.front().frame == WINDOW - 1) {
            Analyze();
        }
    }
}

void FlashWatch::Analyze() {
    const Frame& now = history[CENTER];
    ++checked;
    const auto matches = [this](const Output& output) {
        std::array<const Output*, WINDOW> found{};
        for (size_t f = 0; f < WINDOW; ++f) {
            const auto it = history[f].keys.find(output.key);
            found[f] = it != history[f].keys.end() ? &history[f].outputs[it->second] : nullptr;
        }
        return found;
    };
    const auto has_reference = [](const std::array<const Output*, WINDOW>& found) {
        return (found[0] || found[1]) && (found[3] || found[4]);
    };

    struct Flagged {
        u32 x;
        u32 y;
        double now;
        double reference;
    };
    std::vector<Flagged> flagged;
    const Output* trigger = nullptr;
    // A flicker that alternates every other frame is flagged once, at its start: count each
    // frame brighter than both neighbours too, logged every ALTERNATION_FRAMES.
    bool alternates = false;
    for (const Output& output : now.outputs) {
        if (output.kind != Kind::Present || alternates) {
            continue;
        }
        const auto found = matches(output);
        if (!found[CENTER - 1] || !found[CENTER + 1]) {
            continue;
        }
        u32 tiles = 0;
        for (u32 y = 0; y < output.grid_y && !alternates; ++y) {
            for (u32 x = 0; x < output.grid_x; ++x) {
                std::array<double, 3> mean{};
                for (size_t f = 0; f < 3; ++f) {
                    Stats stats;
                    Add(stats, *found[CENTER - 1 + f], history[CENTER - 1 + f].tiles, x, y);
                    mean[f] = stats.Mean();
                }
                const double neighbours = std::max(mean[0], mean[2]);
                if (mean[1] - neighbours >= 0.5 * delta && mean[1] >= 1.2 * neighbours &&
                    ++tiles >= 2) {
                    alternates = true;
                    break;
                }
            }
        }
    }
    if (alternates) {
        ++alternating;
        ++alternating_window;
    }
    if (now.frame % ALTERNATION_FRAMES == 0) {
        LOG_INFO(Render_Vulkan, "Flash watch: frame {}: {} alternating frames in the last {}, {} in all",
                 now.frame, alternating_window, ALTERNATION_FRAMES, alternating);
        alternating_window = 0;
    }
    for (const Output& output : now.outputs) {
        if (output.kind != Kind::Present) {
            continue;
        }
        const auto found = matches(output);
        if (!has_reference(found)) {
            continue;
        }
        for (u32 y = 0; y < output.grid_y; ++y) {
            for (u32 x = 0; x < output.grid_x; ++x) {
                Window mean;
                Window bad;
                for (size_t f = 0; f < WINDOW; ++f) {
                    if (found[f]) {
                        Stats stats;
                        Add(stats, *found[f], history[f].tiles, x, y);
                        mean[f] = stats.Mean();
                        bad[f] = static_cast<double>(stats.bad);
                    }
                }
                const double mean_reference = *Reference(mean);
                const bool brighter = *mean[CENTER] - mean_reference >= delta &&
                                      *mean[CENTER] >= ratio * mean_reference;
                if (brighter || Spiked(*bad[CENTER], *Reference(bad))) {
                    flagged.push_back({x, y, *mean[CENTER], mean_reference});
                }
            }
        }
        if (!flagged.empty()) {
            trigger = &output;
            break;
        }
    }
    if (!trigger) {
        return;
    }
    ++flashes;

    // The flagged tiles' bounding box with one tile around it.
    u32 min_x = trigger->grid_x;
    u32 min_y = trigger->grid_y;
    u32 max_x = 0;
    u32 max_y = 0;
    for (const Flagged& tile : flagged) {
        min_x = std::min(min_x, tile.x);
        min_y = std::min(min_y, tile.y);
        max_x = std::max(max_x, tile.x);
        max_y = std::max(max_y, tile.y);
    }
    const Region region{
        .x0 = std::max(0.0, (static_cast<double>(min_x) - 1.0) / trigger->grid_x),
        .y0 = std::max(0.0, (static_cast<double>(min_y) - 1.0) / trigger->grid_y),
        .x1 = std::min(1.0, (static_cast<double>(max_x) + 2.0) / trigger->grid_x),
        .y1 = std::min(1.0, (static_cast<double>(max_y) + 2.0) / trigger->grid_y),
    };

    struct Evaluation {
        std::array<const Output*, WINDOW> found{};
        std::array<std::optional<Stats>, WINDOW> region;
        std::array<std::optional<Stats>, WINDOW> whole;
        double mean_ratio = std::numeric_limits<double>::quiet_NaN();
        double peak_ratio = std::numeric_limits<double>::quiet_NaN();
        bool abnormal{};
    };
    std::vector<Evaluation> evaluations(now.outputs.size());
    std::optional<size_t> first;
    size_t abnormal_count = 0;
    for (size_t i = 0; i < now.outputs.size(); ++i) {
        Evaluation& evaluation = evaluations[i];
        evaluation.found = matches(now.outputs[i]);
        Window mean;
        Window peak;
        Window bad;
        Window hot_count;
        Window whole_mean;
        for (size_t f = 0; f < WINDOW; ++f) {
            const Output* const output = evaluation.found[f];
            if (!output) {
                continue;
            }
            const Stats& in_region = evaluation.region[f].emplace(
                Gather(*output, history[f].tiles, region));
            const Stats& whole = evaluation.whole[f].emplace(
                Gather(*output, history[f].tiles, WHOLE));
            mean[f] = in_region.Mean();
            peak[f] = in_region.peak;
            bad[f] = static_cast<double>(in_region.bad);
            hot_count[f] = static_cast<double>(in_region.hot);
            whole_mean[f] = whole.Mean();
        }
        if (!has_reference(evaluation.found)) {
            continue;
        }
        // Dark regions compare against a twentieth of the whole output instead of nothing.
        const double scale = std::max({*Reference(mean), 0.05 * *Reference(whole_mean), 1e-6});
        evaluation.mean_ratio = *mean[CENTER] / scale;
        evaluation.peak_ratio = *peak[CENTER] / std::max(*Reference(peak), 1e-6);
        evaluation.abnormal = evaluation.mean_ratio >= pass_ratio ||
                              Spiked(*bad[CENTER], *Reference(bad)) ||
                              Spiked(*hot_count[CENTER], *Reference(hot_count));
        if (evaluation.abnormal && now.outputs[i].kind != Kind::Present) {
            ++abnormal_count;
            first = first.value_or(i);
        }
    }

    std::string culprit{"none"};
    if (first) {
        const Evaluation& evaluation = evaluations[*first];
        const auto reference = [&evaluation](auto&& get) {
            Window values;
            for (size_t f = 0; f < WINDOW; ++f) {
                if (evaluation.region[f]) {
                    values[f] = get(*evaluation.region[f]);
                }
            }
            return *Reference(values);
        };
        const Stats& in_region = *evaluation.region[CENTER];
        culprit = fmt::format(
            "{} (in the region: mean {:.4g} -> {:.4g}, peak {:.4g} -> {:.4g}, NaN/Inf {} -> {}, "
            "hot {} -> {})",
            Describe(now.outputs[*first]),
            reference([](const Stats& s) { return s.Mean(); }), in_region.Mean(),
            reference([](const Stats& s) { return s.peak; }), in_region.peak,
            reference([](const Stats& s) { return static_cast<double>(s.bad); }), in_region.bad,
            reference([](const Stats& s) { return static_cast<double>(s.hot); }), in_region.hot);
    }
    const bool write = reports < max_reports;
    const std::string name = fmt::format("flash_{:06}", now.frame);
    LOG_WARNING(Render_Vulkan,
                "Flash watch: frame {} ({:.1f} s) flashed in {} presented tiles, region "
                "({:.2f},{:.2f})-({:.2f},{:.2f}); first abnormal output: {}; {} of {} outputs "
                "abnormal; {}",
                now.frame, now.seconds, flagged.size(), region.x0, region.y0, region.x1,
                region.y1, culprit, abnormal_count, now.outputs.size(),
                write ? fmt::format("report {}.jsonl", name)
                      : fmt::format("no report (EDEN_NX2_FLASH_WATCH_MAX={})", max_reports));
    if (!write) {
        return;
    }
    ++reports;

    std::ofstream report{directory / (name + ".jsonl"), std::ios::trunc};
    std::ofstream tiles{directory / (name + ".tiles"), std::ios::binary | std::ios::trunc};
    if (!report || !tiles) {
        LOG_ERROR(Render_Vulkan, "Flash watch: cannot write {}", directory.string());
        return;
    }
    std::string window_frames;
    for (size_t f = 0; f < WINDOW; ++f) {
        window_frames += fmt::format("{}{}", f == 0 ? "" : ",", history[f].frame);
    }
    std::string flagged_json;
    for (const Flagged& tile : flagged) {
        flagged_json += fmt::format("{}[{},{},{},{}]", flagged_json.empty() ? "" : ",", tile.x,
                                    tile.y, JsonNumber(tile.now), JsonNumber(tile.reference));
    }
    report << fmt::format(
                  R"({{"kind":"flash","frame":{},"seconds":{:.3f},"frames":[{}],"trigger":"{}","grid":[{},{}],"region":[{},{},{},{}],"tiles":[{}],"first_abnormal":{},"abnormal":{},"outputs":{},"ratio":{},"delta":{},"pass_ratio":{},"hot":{},"tile_file":"{}.tiles"}})",
                  now.frame, now.seconds, window_frames, trigger->key, trigger->grid_x,
                  trigger->grid_y, JsonNumber(region.x0), JsonNumber(region.y0),
                  JsonNumber(region.x1), JsonNumber(region.y1), flagged_json,
                  first ? fmt::format("{}", *first) : "null", abnormal_count, now.outputs.size(),
                  JsonNumber(ratio), JsonNumber(delta), JsonNumber(pass_ratio), JsonNumber(hot),
                  name)
           << '\n';
    // The tiles file holds, for each output line in order, its tiles in each of the five frames
    // (zeros where the output is missing); "tiles" is the line's first tile.
    u64 tile_offset = 0;
    for (size_t i = 0; i < now.outputs.size(); ++i) {
        const Output& output = now.outputs[i];
        const Evaluation& evaluation = evaluations[i];
        report << fmt::format(
                      R"({{"kind":"output","index":{},"key":"{}","type":"{}","{}":{},"slot":{},"draws":{},"shaders":[{}],"format":"{}","width":{},"height":{},"gpu_addr":"{:#x}","grid":[{},{}],"abnormal":{},"mean_ratio":{},"peak_ratio":{},"region":{},"whole":{},"tiles":{}}})",
                      i, output.key, KindName(output.kind), KindName(output.kind), output.index,
                      output.slot, output.draws, ShaderList(output.shaders, MAX_PASS_SHADERS, true),
                      output.format, output.width, output.height, output.gpu_addr, output.grid_x,
                      output.grid_y, evaluation.abnormal, JsonNumber(evaluation.mean_ratio),
                      JsonNumber(evaluation.peak_ratio), StatsJson(evaluation.region),
                      StatsJson(evaluation.whole), tile_offset)
               << '\n';
        const size_t count = size_t{output.grid_x} * output.grid_y;
        const std::vector<Tile> zeros(count);
        for (size_t f = 0; f < WINDOW; ++f) {
            const Tile* const data = evaluation.found[f]
                                         ? &history[f].tiles[evaluation.found[f]->first_tile]
                                         : zeros.data();
            tiles.write(reinterpret_cast<const char*>(data),
                        static_cast<std::streamsize>(count * sizeof(Tile)));
        }
        tile_offset += count * WINDOW;
    }
}

} // namespace Vulkan
