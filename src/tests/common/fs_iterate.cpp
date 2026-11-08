// SPDX-License-Identifier: GPL-3.0-or-later
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <catch2/catch_test_macros.hpp>
#include "common/fs/fs.h"

namespace {
struct TempTree {
    std::filesystem::path root;
    TempTree() {
        root = std::filesystem::temp_directory_path() /
               ("lichii-fs-iterate-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(root / "sub");
        for (const auto& name : {"a.nca", "b.nca", "c.nca", "sub/d.nca", "sub/e.nca"}) {
            std::ofstream{root / name} << "x";
        }
    }
    ~TempTree() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
};
} // namespace

TEST_CASE("Directory walks stop once the callback asks to", "[fs]") {
    const TempTree tree;
    int visited = 0;
    const Common::FS::DirEntryCallable stop = [&](const std::filesystem::directory_entry&) {
        ++visited;
        return false;
    };
    Common::FS::IterateDirEntries(tree.root, stop, Common::FS::DirEntryFilter::File);
    CHECK(visited == 1);
    visited = 0;
    Common::FS::IterateDirEntriesRecursively(tree.root, stop, Common::FS::DirEntryFilter::File);
    CHECK(visited == 1);
}

TEST_CASE("Directory walks visit every entry while the callback continues", "[fs]") {
    const TempTree tree;
    int visited = 0;
    const Common::FS::DirEntryCallable count = [&](const std::filesystem::directory_entry&) {
        ++visited;
        return true;
    };
    Common::FS::IterateDirEntries(tree.root, count, Common::FS::DirEntryFilter::File);
    CHECK(visited == 3);
    visited = 0;
    Common::FS::IterateDirEntriesRecursively(tree.root, count, Common::FS::DirEntryFilter::File);
    CHECK(visited == 5);
}
