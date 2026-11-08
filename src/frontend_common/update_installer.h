// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// Replaces a portable install in place. Windows lets a running exe and its loaded DLLs be renamed,
// though not deleted, so each file the new package brings is renamed into OldName and the new one
// moved in; the process then restarts. Every rename is written to JournalName first, so a failed or
// interrupted swap is undone, now or at the next start. Only files the package contains are
// touched: the portable `user` folder, nx2_switches.txt and everything else stay as they are.
namespace UpdateInstaller {

// Folder in the install folder for the download, the unpacked package and swaps in progress.
inline constexpr std::string_view StagingName = "update-staging";
// Folder in the install folder with what the last update or rollback replaced.
inline constexpr std::string_view OldName = "update-old";
inline constexpr std::string_view JournalName = "update-journal.txt";
inline constexpr std::string_view LockName = "update.lock";
// In OldName: the files the update added, which a rollback moves out, and the replaced version.
inline constexpr std::string_view AddedName = "update-added.txt";
inline constexpr std::string_view VersionName = "update-version.txt";

// Exclusive lock on a file, held until destruction and released by the system if the process dies.
// It keeps two running copies from swapping or rolling back the same install at once.
class Lock {
public:
    explicit Lock(const std::filesystem::path& path);
    ~Lock();
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

    bool Held() const;

private:
    std::intptr_t handle;
};

// Whether a path relative to the install folder must never be replaced or moved: user data and
// the updater's own files.
bool IsProtected(const std::filesystem::path& relative);

// Whether new files can be created in `dir`.
bool IsWritable(const std::filesystem::path& dir);

// The folder of an unpacked package that holds `exe_name`: `unpacked` itself, or its only
// subfolder (the zip's top folder, such as Fritchii-v5).
std::optional<std::filesystem::path> FindPackageRoot(const std::filesystem::path& unpacked,
                                                     const std::filesystem::path& exe_name);

enum class Outcome {
    Done,
    // Nothing was changed, or every change was undone.
    Unchanged,
    // A change could not be undone; the journal stays so the next start tries again.
    Interrupted,
};

// Install the package folder `package`, inside install_dir/StagingName, over `install_dir`. The
// replaced files go to OldName, which replaces the previous rollback copy, together with
// `replaced_version`. The package must contain `exe_name`, which is swapped last.
Outcome Install(const std::filesystem::path& install_dir, const std::filesystem::path& package,
                std::string_view replaced_version, const std::filesystem::path& exe_name,
                std::error_code& ec);

// The version kept in OldName ("" when unknown), or nullopt when there is nothing to roll back to.
std::optional<std::string> PreviousVersion(const std::filesystem::path& install_dir,
                                           const std::filesystem::path& exe_name);

// Swap OldName back in the same way. The files it replaces become the new OldName, so a rollback
// can itself be rolled back.
Outcome RollBack(const std::filesystem::path& install_dir, std::string_view current_version,
                 const std::filesystem::path& exe_name, std::error_code& ec);

enum class Recovery {
    // No swap was in progress.
    None,
    // The last swap had finished; only its cleanup was left.
    Completed,
    // An interrupted swap was undone.
    Undone,
    Failed,
};

// Finish or undo a swap that a crash or power loss interrupted.
Recovery Recover(const std::filesystem::path& install_dir, std::error_code& ec);

// Remove StagingName, unless a swap still needs it.
void CleanStaging(const std::filesystem::path& install_dir);

// The path of the running exe. Windows reports the new path once the exe is renamed into
// OldName, so callers read it at startup, before any swap.
std::filesystem::path RunningExePath();

// Whether `install_dir` lies in one of the updater's own folders, which are never installed into.
bool IsUpdaterFolder(const std::filesystem::path& install_dir);

// Set for a restarted process to the id of the process it replaces.
inline constexpr char RestartWaitVariable[] = "LICHII_UPDATE_WAIT_PID";

// Wait (up to 30 seconds) for the process named by RestartWaitVariable to exit, so that it has
// saved its settings and closed its log first, then clear the variable.
void WaitForReplacedProcess();

// Command line arguments (without the program) to start the updated version with: the options that
// set up the frontend are kept, the ones that boot a game or run a one-off task are dropped.
std::vector<std::string> RestartArguments(std::span<const std::string> arguments);

} // namespace UpdateInstaller
