// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <catch2/catch_test_macros.hpp>
#include "frontend_common/update_installer.h"

namespace fs = std::filesystem;

namespace {

const fs::path Exe{"lichii.exe"};

class TempDir {
public:
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("lichii-update-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }

    fs::path path;
};

void Write(const fs::path& file, std::string_view text) {
    fs::create_directories(file.parent_path());
    std::ofstream{file, std::ios::binary | std::ios::trunc} << text;
}

std::string Read(const fs::path& file) {
    std::ifstream stream{file, std::ios::binary};
    return {std::istreambuf_iterator<char>{stream}, {}};
}

// An install of version 4 with user data, and the unpacked version 5 package beside it.
struct Install {
    TempDir dir;
    fs::path root = dir.path / "Fritchii";
    fs::path package = root / UpdateInstaller::StagingName / "unpacked" / "Fritchii-v5";

    Install() {
        Write(root / "lichii.exe", "v4");
        Write(root / "Qt6Core.dll", "v4");
        Write(root / "plugins" / "a.dll", "v4");
        Write(root / "user" / "config" / "qt-config.ini", "settings");
        Write(root / "nx2_switches.txt", "switches");
        Write(root / "notes.txt", "mine");

        Write(package / "lichii.exe", "v5");
        Write(package / "Qt6Core.dll", "v5");
        Write(package / "plugins" / "a.dll", "v5");
        Write(package / "plugins" / "new.dll", "v5");
        Write(package / "user" / "config" / "qt-config.ini", "replaced");
        Write(package / "nx2_switches.txt", "replaced");
    }

    void CheckUserData() const {
        CHECK(Read(root / "user" / "config" / "qt-config.ini") == "settings");
        CHECK(Read(root / "nx2_switches.txt") == "switches");
        CHECK(Read(root / "notes.txt") == "mine");
    }
};

} // namespace

TEST_CASE("Updates replace only the package's files and keep one rollback copy", "[updater]") {
    Install install;
    std::error_code ec;
    REQUIRE(UpdateInstaller::Install(install.root, install.package, "4", Exe, ec) ==
            UpdateInstaller::Outcome::Done);
    CHECK_FALSE(ec);

    const auto old_dir = install.root / UpdateInstaller::OldName;
    CHECK(Read(install.root / "lichii.exe") == "v5");
    CHECK(Read(install.root / "Qt6Core.dll") == "v5");
    CHECK(Read(install.root / "plugins" / "new.dll") == "v5");
    CHECK(Read(old_dir / "lichii.exe") == "v4");
    CHECK(Read(old_dir / "plugins" / "a.dll") == "v4");
    CHECK_FALSE(fs::exists(old_dir / "plugins" / "new.dll"));
    CHECK_FALSE(fs::exists(old_dir / "user"));
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::JournalName));
    CHECK(UpdateInstaller::PreviousVersion(install.root, Exe) == "4");
    install.CheckUserData();

    // Rolling back restores version 4, and rolling back again restores version 5.
    REQUIRE(UpdateInstaller::RollBack(install.root, "5", Exe, ec) ==
            UpdateInstaller::Outcome::Done);
    CHECK(Read(install.root / "lichii.exe") == "v4");
    CHECK(Read(install.root / "plugins" / "a.dll") == "v4");
    CHECK_FALSE(fs::exists(install.root / "plugins" / "new.dll"));
    CHECK(UpdateInstaller::PreviousVersion(install.root, Exe) == "5");
    install.CheckUserData();

    REQUIRE(UpdateInstaller::RollBack(install.root, "4", Exe, ec) ==
            UpdateInstaller::Outcome::Done);
    CHECK(Read(install.root / "lichii.exe") == "v5");
    CHECK(Read(install.root / "plugins" / "new.dll") == "v5");
    CHECK(UpdateInstaller::PreviousVersion(install.root, Exe) == "4");
    install.CheckUserData();

    UpdateInstaller::CleanStaging(install.root);
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::StagingName));
}

TEST_CASE("A failed swap puts every file and the old rollback copy back", "[updater]") {
    Install install;
    Write(install.root / UpdateInstaller::OldName / "lichii.exe", "v3");
    Write(install.root / UpdateInstaller::OldName / UpdateInstaller::VersionName, "3");
    // A folder where the package has a file: replacing it fails after Qt6Core.dll was swapped.
    fs::create_directories(install.root / "Qt6Gui.dll");
    Write(install.package / "Qt6Gui.dll", "v5");

    std::error_code ec;
    CHECK(UpdateInstaller::Install(install.root, install.package, "4", Exe, ec) ==
          UpdateInstaller::Outcome::Unchanged);
    CHECK(ec);
    CHECK(Read(install.root / "lichii.exe") == "v4");
    CHECK(Read(install.root / "Qt6Core.dll") == "v4");
    CHECK(Read(install.package / "Qt6Core.dll") == "v5");
    CHECK(fs::is_directory(install.root / "Qt6Gui.dll"));
    CHECK(UpdateInstaller::PreviousVersion(install.root, Exe) == "3");
    CHECK(Read(install.root / UpdateInstaller::OldName / "lichii.exe") == "v3");
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::JournalName));
    install.CheckUserData();
}

TEST_CASE("A package without lichii.exe is refused before anything moves", "[updater]") {
    Install install;
    fs::remove(install.package / "lichii.exe");
    std::error_code ec;
    CHECK(UpdateInstaller::Install(install.root, install.package, "4", Exe, ec) ==
          UpdateInstaller::Outcome::Unchanged);
    CHECK(Read(install.root / "Qt6Core.dll") == "v4");
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::JournalName));
    CHECK_FALSE(UpdateInstaller::PreviousVersion(install.root, Exe));
}

TEST_CASE("A swap interrupted by a crash is undone at the next start", "[updater]") {
    Install install;
    const auto& root = install.root;
    const auto old_dir = root / UpdateInstaller::OldName;
    const auto restore = root / UpdateInstaller::StagingName / "previous-old";
    // The state right after lichii.exe was swapped: the earlier rollback copy is aside in
    // `restore`, the old exe is in OldName, and the journal announces a Qt6Core.dll move that
    // never happened.
    Write(restore / "lichii.exe", "v3");
    fs::create_directories(old_dir);
    fs::rename(root / "lichii.exe", old_dir / "lichii.exe");
    fs::rename(install.package / "lichii.exe", root / "lichii.exe");
    Write(root / UpdateInstaller::JournalName,
          "package update-staging/unpacked/Fritchii-v5\nrestore update-staging/previous-old\n"
          "old lichii.exe\nnew lichii.exe\nold Qt6Core.dll\n");

    std::error_code ec;
    CHECK(UpdateInstaller::Recover(root, ec) == UpdateInstaller::Recovery::Undone);
    CHECK(Read(root / "lichii.exe") == "v4");
    CHECK(Read(root / "Qt6Core.dll") == "v4");
    CHECK(Read(install.package / "lichii.exe") == "v5");
    CHECK(Read(old_dir / "lichii.exe") == "v3");
    CHECK_FALSE(fs::exists(restore));
    CHECK_FALSE(fs::exists(root / UpdateInstaller::JournalName));
    CHECK(UpdateInstaller::Recover(root, ec) == UpdateInstaller::Recovery::None);
    install.CheckUserData();
}

TEST_CASE("A committed swap only has its cleanup finished", "[updater]") {
    Install install;
    const auto restore = install.root / UpdateInstaller::StagingName / "previous-old";
    Write(restore / "lichii.exe", "v3");
    Write(install.root / UpdateInstaller::JournalName,
          "package update-staging/unpacked/Fritchii-v5\nrestore update-staging/previous-old\n"
          "old lichii.exe\nnew lichii.exe\ncommit \n");
    std::error_code ec;
    CHECK(UpdateInstaller::Recover(install.root, ec) == UpdateInstaller::Recovery::Completed);
    CHECK_FALSE(fs::exists(restore));
    CHECK(Read(install.root / "lichii.exe") == "v4");
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::JournalName));
}

TEST_CASE("Journals that point outside the install are not followed", "[updater]") {
    Install install;
    Write(install.root / UpdateInstaller::JournalName,
          "package ../elsewhere\nrestore update-staging/previous-old\nold ../outside.txt\n"
          "new lichii.exe\n");
    std::error_code ec;
    CHECK(UpdateInstaller::Recover(install.root, ec) == UpdateInstaller::Recovery::Failed);
    CHECK(Read(install.root / "lichii.exe") == "v4");
    CHECK(fs::exists(install.root / UpdateInstaller::JournalName));
}

TEST_CASE("A journal cut short in its header is dropped", "[updater]") {
    Install install;
    Write(install.root / UpdateInstaller::JournalName, "package update-staging/unpacked/Fri");
    std::error_code ec;
    CHECK(UpdateInstaller::Recover(install.root, ec) == UpdateInstaller::Recovery::None);
    CHECK_FALSE(fs::exists(install.root / UpdateInstaller::JournalName));
    CHECK(Read(install.root / "lichii.exe") == "v4");
}

TEST_CASE("User data and the updater's own files are protected", "[updater]") {
    for (const auto* path : {"user", "user/config/qt-config.ini", "USER/nand/x", "nx2_switches.txt",
                             "NX2_Switches.TXT", "update-staging/x", "update-old/lichii.exe",
                             "update-journal.txt", "update.lock", "../lichii.exe", "./x", ""}) {
        CHECK(UpdateInstaller::IsProtected(fs::path{path}));
    }
    for (const auto* path : {"lichii.exe", "plugins/a.dll", "users.txt", "plugins/nx2_switches.txt",
                             "plugins/user/a.dll"}) {
        CHECK_FALSE(UpdateInstaller::IsProtected(fs::path{path}));
    }
}

TEST_CASE("Restarts never target the updater's own folders", "[updater]") {
    CHECK(UpdateInstaller::IsUpdaterFolder(fs::path{"C:/Games/Fritchii/update-old"}));
    CHECK(UpdateInstaller::IsUpdaterFolder(fs::path{"C:/Games/Fritchii/Update-Old"}));
    CHECK(UpdateInstaller::IsUpdaterFolder(
        fs::path{"C:/Games/Fritchii/update-staging/unpacked/Fritchii-v5"}));
    CHECK_FALSE(UpdateInstaller::IsUpdaterFolder(fs::path{"C:/Games/Fritchii"}));
    CHECK_FALSE(UpdateInstaller::IsUpdaterFolder(fs::path{"C:/Games/update-older"}));
#if defined(_WIN32) || defined(__linux__)
    CHECK_FALSE(UpdateInstaller::RunningExePath().empty());
#endif
}

TEST_CASE("The package root is the zip's top folder or the unpacked folder", "[updater]") {
    TempDir dir;
    Write(dir.path / "one" / "Fritchii-v5" / "lichii.exe", "v5");
    CHECK(UpdateInstaller::FindPackageRoot(dir.path / "one", Exe) ==
          dir.path / "one" / "Fritchii-v5");
    Write(dir.path / "flat" / "lichii.exe", "v5");
    CHECK(UpdateInstaller::FindPackageRoot(dir.path / "flat", Exe) == dir.path / "flat");
    Write(dir.path / "two" / "a" / "lichii.exe", "v5");
    Write(dir.path / "two" / "b" / "lichii.exe", "v5");
    CHECK_FALSE(UpdateInstaller::FindPackageRoot(dir.path / "two", Exe));
    Write(dir.path / "none" / "a" / "readme.txt", "");
    CHECK_FALSE(UpdateInstaller::FindPackageRoot(dir.path / "none", Exe));
    CHECK(UpdateInstaller::IsWritable(dir.path));
    CHECK_FALSE(UpdateInstaller::IsWritable(dir.path / "missing"));
}

TEST_CASE("Only one updater holds an install's lock", "[updater]") {
    TempDir dir;
    const auto path = dir.path / UpdateInstaller::LockName;
    {
        const UpdateInstaller::Lock first{path};
        const UpdateInstaller::Lock second{path};
        CHECK(first.Held());
        CHECK_FALSE(second.Held());
    }
    const UpdateInstaller::Lock again{path};
    CHECK(again.Held());
}

TEST_CASE("Restarting keeps the frontend options and drops the game", "[updater]") {
    const std::vector<std::string> arguments{"-f",
                                             "-g",
                                             "C:/games/oot.nsp",
                                             "-u",
                                             "1",
                                             "C:/games/other.xci",
                                             "--config=x.ini",
                                             "-qlaunch",
                                             "--pack-nup",
                                             "a",
                                             "-c",
                                             "c.ini",
                                             "--game",
                                             "g.nsp",
                                             "-p",
                                             "args",
                                             "-x",
                                             "*:Info",
                                             "-v",
                                             "--",
                                             "-f"};
    CHECK(
        UpdateInstaller::RestartArguments(arguments) ==
        std::vector<std::string>{"-f", "-u", "1", "--config=x.ini", "-c", "c.ini", "-x", "*:Info"});
    CHECK(UpdateInstaller::RestartArguments({}).empty());
}
