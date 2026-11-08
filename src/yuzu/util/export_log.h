// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

class QWidget;

namespace VkDeviceInfo {
class Record;
}

namespace ExportLog {

/// Asks where to save (Desktop by default), then writes one zip that players can post in a
/// Discord issue: this session's log, the previous one, and a short summary of the PC, the
/// build and the settings that matter for support. The Windows user name and PC name are
/// removed from everything. Shows the zip in Explorer and says in plain words what is inside.
void Export(QWidget* parent, const std::vector<VkDeviceInfo::Record>& vk_device_records);

} // namespace ExportLog
