// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "frontend_common/update_installer.h"

namespace fs = std::filesystem;

namespace {

struct TempDir {
    TempDir() {
        path = fs::temp_directory_path() /
               ("lichii-update-installer-" + std::to_string(std::rand()) + "-" +
                std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    fs::path path;
    static inline int counter = 0;
};

void Write(const fs::path& file, const std::string& content) {
    std::ofstream{file, std::ios::binary} << content;
}

std::string Read(const fs::path& file) {
    std::ifstream stream{file, std::ios::binary};
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("Staged and previous files sit beside the target", "[updater]") {
    const fs::path target = "/opt/lichii/lichii-linux-x86_64.AppImage";
    CHECK(UpdateInstaller::StagedPath(target) == "/opt/lichii/lichii-linux-x86_64.AppImage.new");
    CHECK(UpdateInstaller::PreviousPath(target) == "/opt/lichii/lichii-linux-x86_64.AppImage.prev");
}

TEST_CASE("Swap replaces the target and keeps one previous copy", "[updater]") {
    TempDir dir;
    const auto target = dir.path / "lichii.AppImage";
    const auto staged = UpdateInstaller::StagedPath(target);
    const auto previous = UpdateInstaller::PreviousPath(target);
    Write(target, "old");
    fs::permissions(target, fs::perms::owner_all | fs::perms::group_read);
    Write(staged, "new");
    Write(previous, "stale backup");

    std::error_code ec;
    REQUIRE(UpdateInstaller::CanReplace(target));
    REQUIRE(UpdateInstaller::Swap(staged, target, ec));
    CHECK_FALSE(ec);
    CHECK(Read(target) == "new");
    CHECK(Read(previous) == "old");
    CHECK_FALSE(fs::exists(staged));
    const auto permissions = fs::status(target).permissions();
    CHECK((permissions & fs::perms::owner_exec) != fs::perms::none);
    CHECK((permissions & fs::perms::group_read) != fs::perms::none);
}

TEST_CASE("Swap leaves the target untouched when the staged file is missing", "[updater]") {
    TempDir dir;
    const auto target = dir.path / "lichii.AppImage";
    Write(target, "old");

    std::error_code ec;
    CHECK_FALSE(UpdateInstaller::Swap(UpdateInstaller::StagedPath(target), target, ec));
    CHECK(ec);
    CHECK(Read(target) == "old");
    CHECK_FALSE(fs::exists(UpdateInstaller::PreviousPath(target)));
}

TEST_CASE("Rollback exchanges the target with its previous copy", "[updater]") {
    TempDir dir;
    const auto target = dir.path / "lichii.AppImage";
    const auto previous = UpdateInstaller::PreviousPath(target);
    Write(target, "new");
    Write(previous, "old");

    std::error_code ec;
    REQUIRE(UpdateInstaller::Rollback(target, ec));
    CHECK(Read(target) == "old");
    CHECK(Read(previous) == "new");
    REQUIRE(UpdateInstaller::Rollback(target, ec));
    CHECK(Read(target) == "new");
    CHECK(Read(previous) == "old");
    CHECK_FALSE(fs::exists(UpdateInstaller::StagedPath(target)));
}

TEST_CASE("Rollback without a previous copy fails and changes nothing", "[updater]") {
    TempDir dir;
    const auto target = dir.path / "lichii.AppImage";
    Write(target, "current");

    std::error_code ec;
    CHECK_FALSE(UpdateInstaller::Rollback(target, ec));
    CHECK(ec);
    CHECK(Read(target) == "current");
}

TEST_CASE("Only an existing file in a writable folder can be replaced", "[updater]") {
    TempDir dir;
    const auto target = dir.path / "lichii.AppImage";
    CHECK_FALSE(UpdateInstaller::CanReplace(target));
    CHECK_FALSE(UpdateInstaller::CanReplace(dir.path));
    Write(target, "x");
    Write(dir.path / ".lichii-update-write-test", "kept");
    CHECK(UpdateInstaller::CanReplace(target));
    CHECK(Read(dir.path / ".lichii-update-write-test") == "kept");
    std::size_t count = 0;
    for ([[maybe_unused]] const auto& entry : fs::directory_iterator{dir.path}) {
        ++count;
    }
    CHECK(count == 2);
}

TEST_CASE("Only one holder at a time gets the lock", "[updater]") {
    TempDir dir;
    const auto path = UpdateInstaller::LockPath(dir.path / "lichii.AppImage");
    CHECK(path == dir.path / "lichii.AppImage.lock");
    {
        const UpdateInstaller::Lock first{path};
        CHECK(first.Held());
        const UpdateInstaller::Lock second{path};
        CHECK_FALSE(second.Held());
    }
    const UpdateInstaller::Lock third{path};
    CHECK(third.Held());
}

namespace {

void Put(const fs::path& root, const std::string& relative, const std::string& content) {
    fs::create_directories((root / relative).parent_path());
    Write(root / relative, content);
}

std::set<std::string> Files(const fs::path& root) {
    std::set<std::string> files;
    for (const auto& entry : fs::recursive_directory_iterator{root}) {
        if (entry.is_regular_file()) {
            files.insert(fs::relative(entry.path(), root).generic_string());
        }
    }
    return files;
}

} // namespace

TEST_CASE("Install replaces files, adds new ones and keeps the old ones as a backup", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    const auto staging = install / ".update-staging";
    Put(install, "lichii.exe", "old exe");
    Put(install, "Qt6Core.dll", "old qt");
    Put(install, "plugins/a.dll", "old plugin");
    Put(install, "user/config/qt-config.ini", "my settings");
    Put(staging, "lichii.exe", "new exe");
    Put(staging, "Qt6Core.dll", "new qt");
    Put(staging, "plugins/b.dll", "new plugin");
    Put(staging, "user/config/qt-config.ini", "must not be installed");

    std::error_code ec;
    REQUIRE(UpdateInstaller::Install(staging, install, ec));

    CHECK(Read(install / "lichii.exe") == "new exe");
    CHECK(Read(install / "Qt6Core.dll") == "new qt");
    CHECK(Read(install / "plugins/b.dll") == "new plugin");
    CHECK(Read(install / "plugins/a.dll") == "old plugin");
    CHECK(Read(install / "user/config/qt-config.ini") == "my settings");
    CHECK_FALSE(fs::exists(staging));
    const auto backup = install / ".update-backup";
    CHECK(Read(backup / "lichii.exe") == "old exe");
    CHECK(Read(backup / "Qt6Core.dll") == "old qt");
    CHECK_FALSE(fs::exists(backup / "user"));
    CHECK(Files(install / "user") == std::set<std::string>{"config/qt-config.ini"});
    CHECK(Read(install / "install-manifest.txt") ==
          "Qt6Core.dll\nlichii.exe\nplugins/b.dll\n");
    CHECK_FALSE(fs::exists(install / ".update-backup.tmp"));
}

TEST_CASE("A later update removes the files the new version dropped", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    Put(install, "lichii.exe", "v1");
    Put(install, "old.dll", "v1 only");
    Put(install, "keep.dll", "v1");
    Put(install, "user/data.bin", "mine");
    std::error_code ec;

    Put(install, ".update-staging/lichii.exe", "v2");
    Put(install, ".update-staging/old.dll", "v2");
    Put(install, ".update-staging/keep.dll", "v2");
    REQUIRE(UpdateInstaller::Install(install / ".update-staging", install, ec));

    Put(install, ".update-staging/lichii.exe", "v3");
    Put(install, ".update-staging/keep.dll", "v3");
    REQUIRE(UpdateInstaller::Install(install / ".update-staging", install, ec));

    CHECK(Files(install) ==
          std::set<std::string>{"lichii.exe", "keep.dll", "install-manifest.txt",
                                ".update-backup/lichii.exe", ".update-backup/keep.dll",
                                ".update-backup/old.dll", ".update-backup/install-manifest.txt",
                                "user/data.bin"});
    CHECK(Read(install / "lichii.exe") == "v3");
    CHECK(Read(install / ".update-backup/old.dll") == "v2");
    CHECK(Read(install / ".update-backup/lichii.exe") == "v2");
}

TEST_CASE("Restoring the backup brings back the previous version and can be undone", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    Put(install, "lichii.exe", "v1");
    Put(install, "gone.dll", "v1 only");
    Put(install, "user/data.bin", "mine");
    std::error_code ec;

    Put(install, ".update-staging/lichii.exe", "v2");
    Put(install, ".update-staging/new.dll", "v2 only");
    REQUIRE(UpdateInstaller::Install(install / ".update-staging", install, ec));
    REQUIRE(fs::exists(install / "new.dll"));

    REQUIRE(UpdateInstaller::Install(install / ".update-backup", install, ec));
    CHECK(Read(install / "lichii.exe") == "v1");
    CHECK(Read(install / "gone.dll") == "v1 only");
    CHECK_FALSE(fs::exists(install / "new.dll"));
    CHECK(Read(install / "user/data.bin") == "mine");
    CHECK(Read(install / ".update-backup/lichii.exe") == "v2");
    CHECK(Read(install / ".update-backup/new.dll") == "v2 only");

    REQUIRE(UpdateInstaller::Install(install / ".update-backup", install, ec));
    CHECK(Read(install / "lichii.exe") == "v2");
    CHECK(Read(install / "new.dll") == "v2 only");
    CHECK_FALSE(fs::exists(install / "gone.dll"));
    CHECK(Read(install / "user/data.bin") == "mine");
}

#ifndef _WIN32
TEST_CASE("A failed install puts every file back", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    Put(install, "a.dll", "old a");
    Put(install, "locked/b.dll", "old b");
    Put(install, ".update-staging/a.dll", "new a");
    Put(install, ".update-staging/locked/b.dll", "new b");
    if (geteuid() == 0) {
        SKIP("root ignores directory permissions");
    }
    fs::permissions(install / "locked", fs::perms::owner_read | fs::perms::owner_exec);

    std::error_code ec;
    CHECK_FALSE(UpdateInstaller::Install(install / ".update-staging", install, ec));
    CHECK(ec);
    fs::permissions(install / "locked", fs::perms::owner_all);
    CHECK(Read(install / "a.dll") == "old a");
    CHECK(Read(install / "locked/b.dll") == "old b");
    CHECK_FALSE(fs::exists(install / ".update-backup"));
    CHECK_FALSE(fs::exists(install / ".update-backup.tmp"));
    CHECK_FALSE(fs::exists(install / "install-manifest.txt"));
}

TEST_CASE("A failed restore keeps the backup it restores from", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    const auto backup = install / ".update-backup";
    Put(install, "a.dll", "v2 a");
    Put(install, "locked/b.dll", "v2 b");
    Put(backup, "a.dll", "v1 a");
    Put(backup, "locked/b.dll", "v1 b");
    if (geteuid() == 0) {
        SKIP("root ignores directory permissions");
    }
    fs::permissions(install / "locked", fs::perms::owner_read | fs::perms::owner_exec);

    std::error_code ec;
    CHECK_FALSE(UpdateInstaller::Install(backup, install, ec));
    fs::permissions(install / "locked", fs::perms::owner_all);
    CHECK(Read(install / "a.dll") == "v2 a");
    CHECK(Read(install / "locked/b.dll") == "v2 b");
    CHECK(Read(backup / "a.dll") == "v1 a");
    CHECK(Read(backup / "locked/b.dll") == "v1 b");
}
#endif

TEST_CASE("The user folder is protected whatever its letter case", "[updater]") {
    TempDir dir;
    const auto install = dir.path / "app";
    const auto staging = install / ".update-staging";
    Put(install, "lichii.exe", "old exe");
    Put(install, "USER/data.bin", "mine");
    Put(staging, "lichii.exe", "new exe");
    Put(staging, "User/config.ini", "must not be installed");

    std::error_code ec;
    REQUIRE(UpdateInstaller::Install(staging, install, ec));
    CHECK(Read(install / "USER/data.bin") == "mine");
    CHECK_FALSE(fs::exists(install / "User/config.ini"));
    CHECK(Read(install / "install-manifest.txt") == "lichii.exe\n");
    CHECK(Files(install / ".update-backup") ==
          std::set<std::string>{"lichii.exe", "install-manifest.txt"});
}

TEST_CASE("Path list entries inside a folder are removed", "[updater]") {
    using UpdateInstaller::StripPathListEntries;
    CHECK(StripPathListEntries("/tmp/.mount_x/usr/lib:/usr/lib:/tmp/.mount_x:/opt/lib",
                               "/tmp/.mount_x") == "/usr/lib:/opt/lib");
    CHECK(StripPathListEntries("/tmp/.mount_xy/lib:/usr/lib", "/tmp/.mount_x") ==
          "/tmp/.mount_xy/lib:/usr/lib");
    CHECK(StripPathListEntries("", "/tmp/.mount_x").empty());
    CHECK(StripPathListEntries("/a::/b", "/tmp/.mount_x") == "/a:/b");
    CHECK(StripPathListEntries("/a:/b", "") == "/a:/b");
}
