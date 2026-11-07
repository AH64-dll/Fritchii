// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

namespace Vulkan::Detail {

inline bool IsOverwolfLayer(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text.find("overwolf") != std::string::npos ||
           text.find("ow-graphics") != std::string::npos;
}

inline void CollectOverwolfLayerControls(const nlohmann::json& manifest,
                                         const std::string& manifest_path,
                                         std::set<std::string>& layer_names,
                                         std::map<std::string, std::string>& environment) {
    const auto collect = [&](const nlohmann::json& layer) {
        if (!layer.is_object()) {
            return;
        }
        const auto name = layer.find("name");
        const auto library = layer.find("library_path");
        if (!IsOverwolfLayer(manifest_path) &&
            !(name != layer.end() && name->is_string() &&
              IsOverwolfLayer(name->get<std::string>())) &&
            !(library != layer.end() && library->is_string() &&
              IsOverwolfLayer(library->get<std::string>()))) {
            return;
        }
        if (name != layer.end() && name->is_string() &&
            !name->get_ref<const std::string&>().empty()) {
            layer_names.insert(name->get<std::string>());
        }
        const auto disable = layer.find("disable_environment");
        if (disable == layer.end() || !disable->is_object()) {
            return;
        }
        for (const auto& [key, value] : disable->items()) {
            if (!key.empty() && key.find('=') == std::string::npos && value.is_string()) {
                // The loader requires a non-empty value even when the manifest specifies "".
                const auto text = value.get<std::string>();
                environment.insert_or_assign(key, text.empty() ? "1" : text);
            }
        }
    };
    if (!manifest.is_object()) {
        return;
    }
    if (const auto layer = manifest.find("layer"); layer != manifest.end()) {
        collect(*layer);
    }
    if (const auto layers = manifest.find("layers");
        layers != manifest.end() && layers->is_array()) {
        for (const auto& layer : *layers) {
            collect(layer);
        }
    }
}

} // namespace Vulkan::Detail
