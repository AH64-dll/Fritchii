// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/vulkan_overlay_filter.h"

TEST_CASE("Overwolf Vulkan manifests supply exact disable controls", "[video_core]") {
    std::set<std::string> names;
    std::map<std::string, std::string> environment;
    const auto manifest = nlohmann::json::parse(R"({"layer": {
        "name": "VK_LAYER_OW_OBS_HOOK", "library_path": "ow-graphics-hook64.dll",
        "disable_environment": {"DISABLE_OW_TEST": "1"}
    }})");
    Vulkan::Detail::CollectOverwolfLayerControls(manifest, "C:/layers/overlay.json", names,
                                                 environment);
    REQUIRE(names.contains("VK_LAYER_OW_OBS_HOOK"));
    REQUIRE(environment.at("DISABLE_OW_TEST") == "1");

    const auto unrelated = nlohmann::json::parse(R"({"layer": {
        "name": "VK_LAYER_OBS_HOOK", "library_path": "graphics-hook64.dll",
        "disable_environment": {"DISABLE_OBS_TEST": "1"}
    }})");
    Vulkan::Detail::CollectOverwolfLayerControls(unrelated, "C:/OBS/overlay.json", names,
                                                 environment);
    REQUIRE(names.size() == 1);
    REQUIRE_FALSE(environment.contains("DISABLE_OBS_TEST"));
}

TEST_CASE("Overwolf manifest paths and malformed fields are handled", "[video_core]") {
    std::set<std::string> names;
    std::map<std::string, std::string> environment;
    const auto manifest = nlohmann::json::parse(R"({"layers": [
        {"name": "VK_LAYER_CAPTURE", "disable_environment": {"DISABLE_CAPTURE_TEST": ""}},
        {"name": 42, "disable_environment": false},
        {"name": "VK_LAYER_OTHER", "disable_environment": {"BAD=NAME": "1", "BAD_TYPE": 2}}
    ]})");
    Vulkan::Detail::CollectOverwolfLayerControls(manifest, "C:/OVERWOLF/layers.json", names,
                                                 environment);
    REQUIRE(names.size() == 2);
    REQUIRE(environment.at("DISABLE_CAPTURE_TEST") == "1");
    REQUIRE(environment.size() == 1);
    Vulkan::Detail::CollectOverwolfLayerControls(nlohmann::json::parse("{invalid", nullptr, false),
                                                 "invalid.json", names, environment);
    Vulkan::Detail::CollectOverwolfLayerControls(manifest, "C:/OVERWOLF/layers.json", names,
                                                 environment);
    REQUIRE(names.size() == 2);
    REQUIRE(environment.size() == 1);
}
