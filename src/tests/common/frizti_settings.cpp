// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "common/frizti_settings.h"

namespace Frizti = Settings::Frizti;

namespace {
// What another build may leave in the shared config.
void SetOtherBuildValues(Settings::Values& values) {
    values.gpu_accuracy.SetValue(Settings::GpuAccuracy::High);
    values.dma_accuracy.SetValue(Settings::DmaAccuracy::Safe);
    values.gpu_fence_behavior.SetValue(Settings::GpuFenceBehavior::Accurate);
    values.vsync_mode.SetValue(Settings::VSyncMode::Fifo);
    values.async_presentation.SetValue(false);
    values.use_asynchronous_shaders.SetValue(false);
    values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
    values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_4Gb);
    values.nvdec_emulation.SetValue(Settings::NvdecEmulation::Cpu);
    values.scaling_filter.SetValue(Settings::ScalingFilter::Fsr3);
}

void CheckFriztiValues(const Settings::Values& values) {
    CHECK(values.gpu_accuracy.GetValue() == Settings::GpuAccuracy::Low);
    CHECK(values.dma_accuracy.GetValue() == Settings::DmaAccuracy::Default);
    CHECK(values.gpu_fence_behavior.GetValue() == Settings::GpuFenceBehavior::Default);
    CHECK(values.vsync_mode.GetValue() == Settings::VSyncMode::Mailbox);
    CHECK(values.async_presentation.GetValue());
    CHECK(values.use_asynchronous_shaders.GetValue());
    CHECK(values.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked);
    CHECK(values.memory_layout_mode.GetValue() == Settings::MemoryLayout::Memory_8Gb);
    CHECK(values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu);
    CHECK(values.scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
}
} // namespace

TEST_CASE("Frizti's values are the tuned settings' defaults", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    CheckFriztiValues(*values);
    for (const auto* setting : Frizti::TunedSettings(*values)) {
        CHECK(setting->Switchable());
    }
}

TEST_CASE("A new revision sets every tuned setting", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    SetOtherBuildValues(*values);
    const auto tuned = Frizti::TunedSettings(*values);
    const auto changes = Frizti::ReconcileGlobal(tuned, {}, true);
    CHECK(changes.size() == tuned.size());
    CheckFriztiValues(*values);
    CHECK(Frizti::ReconcileGlobal(tuned, {}, true).empty());
}

TEST_CASE("Between revisions only another build's changes are undone", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    const auto tuned = Frizti::TunedSettings(*values);
    // The player chose High in Frizti, and Frizti wrote it.
    values->gpu_accuracy.SetValue(Settings::GpuAccuracy::High);
    Frizti::Record record;
    record.revision = Frizti::Revision;
    record.global = Frizti::GlobalValues(tuned);

    // Another build then saved FIFO and FSR.
    values->vsync_mode.SetValue(Settings::VSyncMode::Fifo);
    values->scaling_filter.SetValue(Settings::ScalingFilter::Fsr3);
    const auto changes = Frizti::ReconcileGlobal(tuned, record, false);
    REQUIRE(changes.size() == 2);
    CHECK(values->gpu_accuracy.GetValue() == Settings::GpuAccuracy::High);
    CHECK(values->vsync_mode.GetValue() == Settings::VSyncMode::Mailbox);
    CHECK(values->scaling_filter.GetValue() == Settings::ScalingFilter::ScaleForce);
    CHECK(Frizti::FormatChanges(changes) ==
          "use_vsync: Fifo -> Mailbox, scaling_filter: Fsr3 -> ScaleForce");
}

TEST_CASE("Ocarina's per-game overrides go back to global", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    const auto tuned = Frizti::TunedSettings(*values);
    const auto override_value = [](auto& setting, auto value) {
        setting.SetGlobal(false);
        setting.SetValue(value);
    };
    override_value(values->gpu_accuracy, Settings::GpuAccuracy::High);
    override_value(values->vsync_mode, Settings::VSyncMode::Fifo);

    SECTION("a new revision drops every override") {
        const auto changes = Frizti::ReconcilePerGame(tuned, {}, true);
        CHECK(changes.size() == 2);
        CHECK(values->gpu_accuracy.UsingGlobal());
        CHECK(values->vsync_mode.UsingGlobal());
        CHECK(changes[0].label == "0400486024DEC000 gpu_accuracy");
        CHECK(changes[0].to == "global");
    }

    SECTION("overrides Frizti wrote are kept between revisions") {
        Frizti::Record record;
        record.per_game = Frizti::PerGameValues(tuned);
        override_value(values->memory_layout_mode, Settings::MemoryLayout::Memory_4Gb);
        const auto changes = Frizti::ReconcilePerGame(tuned, record, false);
        REQUIRE(changes.size() == 1);
        CHECK(changes[0].label == "0400486024DEC000 memory_layout_mode");
        CHECK_FALSE(values->gpu_accuracy.UsingGlobal());
        CHECK_FALSE(values->vsync_mode.UsingGlobal());
        CHECK(values->memory_layout_mode.UsingGlobal());
    }

    for (const auto& restore : values->linkage.restore_functions) {
        restore();
    }
}

TEST_CASE("Docked mode replaces the handheld controller", "[frizti_settings]") {
    auto values = std::make_unique<Settings::Values>();
    values->players.GetValue(true)[0].controller_type = Settings::ControllerType::Handheld;
    CHECK(Frizti::FixDockedController(*values).size() == 1);
    CHECK(values->players.GetValue(true)[0].controller_type ==
          Settings::ControllerType::ProController);
    CHECK(Frizti::FixDockedController(*values).empty());
}

TEST_CASE("The Frizti record round-trips", "[frizti_settings]") {
    Frizti::Record record;
    record.revision = 2;
    record.global = {{"gpu_accuracy", "0"}, {"use_vsync", "1"}};
    record.per_game = {{"gpu_accuracy", "global"}, {"memory_layout_mode", "1"}};
    const auto parsed = Frizti::ParseRecord(Frizti::SerializeRecord(record));
    CHECK(parsed.revision == record.revision);
    CHECK(parsed.global == record.global);
    CHECK(parsed.per_game == record.per_game);

    const auto empty = Frizti::ParseRecord("");
    CHECK(empty.revision == 0);
    CHECK(empty.global.empty());
    CHECK(Frizti::ParseRecord("revision=x\r\nglobal.a = 1\r\n").global.at("a") == "1");
}
