// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "frontend_common/update_installer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace UpdateInstaller {

namespace {

std::string ToUtf8(const fs::path& path) {
    const auto text = path.generic_u8string();
    return {text.begin(), text.end()};
}

fs::path FromUtf8(std::string_view text) {
    return fs::path{std::u8string{text.begin(), text.end()}};
}

bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
    return std::ranges::equal(
        a, b, [](unsigned char x, unsigned char y) { return std::tolower(x) == std::tolower(y); });
}

bool Exists(const fs::path& path) {
    std::error_code ec;
    return fs::symlink_status(path, ec).type() != fs::file_type::not_found && !ec;
}

// A path that stays inside the folder it is relative to.
bool IsContained(const fs::path& relative) {
    return !relative.empty() && relative.is_relative() && !relative.has_root_name() &&
           std::ranges::none_of(relative,
                                [](const fs::path& part) { return part == ".." || part == "."; });
}

// Rename without replacing anything. A rename can fail for a moment on Windows while a virus
// scanner or the search indexer holds the file, so it is retried for about two seconds.
void Move(const fs::path& from, const fs::path& to, std::error_code& ec) {
    if (Exists(to)) {
        ec = std::make_error_code(std::errc::file_exists);
        return;
    }
    fs::create_directories(to.parent_path(), ec);
    if (ec) {
        return;
    }
    for (int attempt = 0; attempt < 20; ++attempt) {
        fs::rename(from, to, ec);
        if (!ec || !Exists(from)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool IsFile(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

bool IsMetadata(const fs::path& relative) {
    return relative == fs::path{AddedName} || relative == fs::path{VersionName};
}

// Whether `dir` holds files other than its metadata.
bool HasPayload(const fs::path& dir) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it{dir, ec}, end; !ec && it != end; it.increment(ec)) {
        if (!it->is_directory(ec) && !IsMetadata(it->path().lexically_relative(dir))) {
            return true;
        }
    }
    return static_cast<bool>(ec);
}

std::vector<fs::path> ReadList(const fs::path& file) {
    std::vector<fs::path> list;
    std::ifstream stream{file, std::ios::binary};
    for (std::string line; std::getline(stream, line);) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (const auto path = FromUtf8(line); IsContained(path)) {
            list.push_back(path);
        }
    }
    return list;
}

bool WriteText(const fs::path& file, std::string_view text) {
    std::ofstream stream{file, std::ios::binary | std::ios::trunc};
    stream << text;
    stream.flush();
    return stream.good();
}

class Journal {
public:
    explicit Journal(const fs::path& path) : stream{path, std::ios::binary | std::ios::trunc} {}

    // The line reaches the OS before the step it announces is taken.
    bool Write(std::string_view kind, const fs::path& path) {
        stream << kind << ' ' << ToUtf8(path) << '\n';
        stream.flush();
        return stream.good();
    }

private:
    std::ofstream stream;
};

// Move the files of `package` over `install_dir`, and the files in `obsolete` out of it, keeping
// what they replace in OldName. The current OldName first moves to `restore`, which becomes OldName
// again if the swap is undone and is deleted once it is done.
Outcome Swap(const fs::path& install_dir, const fs::path& package, const fs::path& restore,
             const std::vector<fs::path>& obsolete, std::string_view replaced_version,
             const fs::path& exe_name, std::error_code& ec) {
    const auto old_dir = install_dir / OldName;
    const auto journal_path = install_dir / JournalName;
    if (Exists(journal_path)) {
        ec = std::make_error_code(std::errc::device_or_resource_busy);
        return Outcome::Unchanged;
    }
    const auto package_relative = package.lexically_relative(install_dir);
    const auto restore_relative = restore.lexically_relative(install_dir);
    if (!IsContained(package_relative) || !IsContained(restore_relative)) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return Outcome::Unchanged;
    }

    // A leftover from an earlier failure would otherwise be restored as the rollback copy.
    if (Exists(restore)) {
        fs::remove_all(restore, ec);
        if (ec) {
            return Outcome::Unchanged;
        }
    }

    std::optional<Journal> journal{std::in_place, journal_path};
    // Recover reads and removes the journal, so it is closed first.
    const auto undo = [&] {
        journal.reset();
        std::error_code recover_ec;
        const auto recovery = Recover(install_dir, recover_ec);
        return recovery == Recovery::Failed ? Outcome::Interrupted : Outcome::Unchanged;
    };

    if (!journal->Write("package", package_relative) ||
        !journal->Write("restore", restore_relative)) {
        ec = std::make_error_code(std::errc::io_error);
        return undo();
    }
    if (Exists(old_dir)) {
        Move(old_dir, restore, ec);
        if (ec) {
            return undo();
        }
    }

    std::vector<fs::path> files;
    for (fs::recursive_directory_iterator it{package, ec}, end; !ec && it != end;
         it.increment(ec)) {
        const auto relative = it->path().lexically_relative(package);
        if (it->is_symlink(ec) || !it->is_regular_file(ec) || IsMetadata(relative) ||
            IsProtected(relative)) {
            continue;
        }
        files.push_back(relative);
    }
    const auto is_exe = [&](const fs::path& relative) {
        return EqualsIgnoreCase(ToUtf8(relative), ToUtf8(exe_name));
    };
    if (!ec && std::ranges::none_of(files, is_exe)) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
    }
    if (ec) {
        return undo();
    }
    std::ranges::sort(files, [&](const fs::path& a, const fs::path& b) {
        const bool a_exe = is_exe(a);
        const bool b_exe = is_exe(b);
        return a_exe != b_exe ? b_exe : a < b;
    });

    fs::create_directories(old_dir, ec);
    if (ec || !WriteText(old_dir / VersionName, replaced_version)) {
        ec = ec ? ec : std::make_error_code(std::errc::io_error);
        return undo();
    }

    std::string added;
    for (const auto& relative : files) {
        const auto target = install_dir / relative;
        if (Exists(target)) {
            if (!IsFile(target) || !journal->Write("old", relative)) {
                ec = std::make_error_code(std::errc::io_error);
                return undo();
            }
            Move(target, old_dir / relative, ec);
            if (ec) {
                return undo();
            }
        } else {
            added += ToUtf8(relative) + '\n';
        }
        if (!journal->Write("new", relative)) {
            ec = std::make_error_code(std::errc::io_error);
            return undo();
        }
        Move(package / relative, target, ec);
        if (ec) {
            return undo();
        }
    }

    const std::set<fs::path> placed{files.begin(), files.end()};
    for (const auto& relative : obsolete) {
        const auto target = install_dir / relative;
        if (placed.contains(relative) || IsProtected(relative) || !IsFile(target)) {
            continue;
        }
        if (!journal->Write("old", relative)) {
            ec = std::make_error_code(std::errc::io_error);
            return undo();
        }
        Move(target, old_dir / relative, ec);
        if (ec) {
            return undo();
        }
    }

    if (!WriteText(old_dir / AddedName, added) || !journal->Write("commit", {})) {
        ec = std::make_error_code(std::errc::io_error);
        return undo();
    }

    // Done: what is left is cleanup, which the next start finishes if it fails here.
    journal.reset();
    std::error_code cleanup_ec;
    fs::remove_all(restore, cleanup_ec);
    fs::remove(journal_path, cleanup_ec);
    ec.clear();
    return Outcome::Done;
}

} // namespace

#ifdef _WIN32
Lock::Lock(const fs::path& path)
    : handle(reinterpret_cast<std::intptr_t>(
          CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr))) {}

Lock::~Lock() {
    if (Held()) {
        CloseHandle(reinterpret_cast<HANDLE>(handle));
    }
}

bool Lock::Held() const {
    return reinterpret_cast<HANDLE>(handle) != INVALID_HANDLE_VALUE;
}
#else
Lock::Lock(const fs::path& path) : handle(open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600)) {
    if (handle >= 0 && flock(static_cast<int>(handle), LOCK_EX | LOCK_NB) != 0) {
        close(static_cast<int>(handle));
        handle = -1;
    }
}

Lock::~Lock() {
    if (Held()) {
        close(static_cast<int>(handle));
    }
}

bool Lock::Held() const {
    return handle >= 0;
}
#endif

bool IsProtected(const fs::path& relative) {
    if (!IsContained(relative)) {
        return true;
    }
    const auto first = ToUtf8(*relative.begin());
    for (const auto name : {std::string_view{"user"}, StagingName, OldName}) {
        if (EqualsIgnoreCase(first, name)) {
            return true;
        }
    }
    if (std::distance(relative.begin(), relative.end()) == 1) {
        for (const auto name : {std::string_view{"nx2_switches.txt"}, JournalName, LockName}) {
            if (EqualsIgnoreCase(first, name)) {
                return true;
            }
        }
    }
    return false;
}

bool IsWritable(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return false;
    }
    const auto probe =
        dir /
        ("update-write-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp");
    if (Exists(probe)) {
        return false;
    }
    const bool writable = std::ofstream{probe}.is_open();
    fs::remove(probe, ec);
    return writable;
}

std::optional<fs::path> FindPackageRoot(const fs::path& unpacked, const fs::path& exe_name) {
    std::error_code ec;
    if (fs::is_regular_file(unpacked / exe_name, ec)) {
        return unpacked;
    }
    std::optional<fs::path> only;
    for (fs::directory_iterator it{unpacked, ec}, end; !ec && it != end; it.increment(ec)) {
        if (only || !it->is_directory(ec)) {
            return std::nullopt;
        }
        only = it->path();
    }
    if (ec || !only || !fs::is_regular_file(*only / exe_name, ec)) {
        return std::nullopt;
    }
    return only;
}

Outcome Install(const fs::path& install_dir, const fs::path& package,
                std::string_view replaced_version, const fs::path& exe_name, std::error_code& ec) {
    return Swap(install_dir, package, install_dir / StagingName / "previous-old", {},
                replaced_version, exe_name, ec);
}

std::optional<std::string> PreviousVersion(const fs::path& install_dir, const fs::path& exe_name) {
    const auto old_dir = install_dir / OldName;
    std::error_code ec;
    if (!fs::is_regular_file(old_dir / exe_name, ec)) {
        return std::nullopt;
    }
    std::ifstream stream{old_dir / VersionName, std::ios::binary};
    std::string version;
    std::getline(stream, version);
    return version;
}

Outcome RollBack(const fs::path& install_dir, std::string_view current_version,
                 const fs::path& exe_name, std::error_code& ec) {
    if (!PreviousVersion(install_dir, exe_name)) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return Outcome::Unchanged;
    }
    // OldName itself becomes the package once Swap has moved it aside.
    const auto package = install_dir / StagingName / "rollback";
    return Swap(install_dir, package, package, ReadList(install_dir / OldName / AddedName),
                current_version, exe_name, ec);
}

Recovery Recover(const fs::path& install_dir, std::error_code& ec) {
    const auto journal_path = install_dir / JournalName;
    if (!Exists(journal_path)) {
        return Recovery::None;
    }

    fs::path package;
    fs::path restore;
    std::vector<std::pair<bool, fs::path>> steps;
    bool committed = false;
    {
        std::ifstream stream{journal_path, std::ios::binary};
        for (std::string line; std::getline(stream, line);) {
            const auto space = line.find(' ');
            const std::string_view kind = std::string_view{line}.substr(0, space);
            const auto path =
                space == std::string::npos ? fs::path{} : FromUtf8(line.substr(space + 1));
            if (kind == "commit") {
                committed = true;
            } else if (!IsContained(path)) {
                // A line cut short by a power loss announces a step that was never taken.
                continue;
            } else if (kind == "package") {
                package = install_dir / path;
            } else if (kind == "restore") {
                restore = install_dir / path;
            } else if (kind == "old" || kind == "new") {
                steps.emplace_back(kind == "new", path);
            }
        }
    }

    const auto old_dir = install_dir / OldName;
    if (committed) {
        if (!restore.empty()) {
            fs::remove_all(restore, ec);
        }
        fs::remove(journal_path, ec);
        ec.clear();
        return Recovery::Completed;
    }
    if (package.empty() || restore.empty()) {
        // Cut short while the header was written: nothing has moved yet.
        if (steps.empty() && (restore.empty() || !Exists(restore))) {
            fs::remove(journal_path, ec);
            return ec ? Recovery::Failed : Recovery::None;
        }
        ec = std::make_error_code(std::errc::invalid_argument);
        return Recovery::Failed;
    }

    bool failed = false;
    for (auto step = steps.rbegin(); step != steps.rend(); ++step) {
        const auto& [placed_new, relative] = *step;
        const auto installed = install_dir / relative;
        const auto away = placed_new ? package / relative : old_dir / relative;
        std::error_code step_ec;
        if (placed_new) {
            // The new file is either still in the package or already installed.
            if (Exists(installed) && !Exists(away)) {
                Move(installed, away, step_ec);
            }
        } else if (Exists(away)) {
            if (Exists(installed)) {
                step_ec = std::make_error_code(std::errc::file_exists);
            } else {
                Move(away, installed, step_ec);
            }
        }
        if (step_ec) {
            failed = true;
            ec = step_ec;
        }
    }
    if (failed) {
        return Recovery::Failed;
    }

    // Put the previous rollback copy back in place of the one this swap started.
    if (Exists(restore)) {
        if (Exists(old_dir)) {
            if (HasPayload(old_dir)) {
                ec = std::make_error_code(std::errc::directory_not_empty);
                return Recovery::Failed;
            }
            fs::remove_all(old_dir, ec);
            if (ec) {
                return Recovery::Failed;
            }
        }
        Move(restore, old_dir, ec);
        if (ec) {
            return Recovery::Failed;
        }
    } else if (Exists(old_dir) && !HasPayload(old_dir)) {
        std::error_code cleanup_ec;
        fs::remove_all(old_dir, cleanup_ec);
    }
    fs::remove(journal_path, ec);
    return ec ? Recovery::Failed : Recovery::Undone;
}

void CleanStaging(const fs::path& install_dir) {
    if (Exists(install_dir / JournalName)) {
        return;
    }
    std::error_code ec;
    fs::remove_all(install_dir / StagingName, ec);
}

fs::path RunningExePath() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const auto size =
            GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (size == 0) {
            return {};
        }
        if (size < buffer.size()) {
            buffer.resize(size);
            return fs::path{buffer};
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    std::error_code ec;
    auto path = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path{} : path;
#endif
}

bool IsUpdaterFolder(const fs::path& install_dir) {
    return std::ranges::any_of(install_dir, [](const fs::path& part) {
        const auto name = ToUtf8(part);
        return EqualsIgnoreCase(name, OldName) || EqualsIgnoreCase(name, StagingName);
    });
}

void WaitForReplacedProcess() {
    const char* value = std::getenv(RestartWaitVariable);
    if (value == nullptr || *value == '\0') {
        return;
    }
    const auto pid = std::strtoul(value, nullptr, 10);
#ifdef _WIN32
    _putenv_s(RestartWaitVariable, "");
    if (const HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid))) {
        WaitForSingleObject(process, 30000);
        CloseHandle(process);
    }
#else
    unsetenv(RestartWaitVariable);
    for (int i = 0; i < 300 && pid != 0 && kill(static_cast<pid_t>(pid), 0) == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
}

std::vector<std::string> RestartArguments(std::span<const std::string> arguments) {
    struct Option {
        std::string_view short_name;
        std::string_view long_name;
        bool value;
        bool keep;
    };
    // The options of Core::ParseLaunchParams.
    static constexpr Option options[] = {
        {"-c", "--config", true, true},        {"-u", "--user", true, true},
        {"-i", "--input-profile", true, true}, {"-d", "--debug", true, true},
        {"-x", "--filter", true, true},        {"-f", "--fullscreen", false, true},
        {"-n", "--null-render", false, true},  {"-s", "--singlecore", false, true},
        {"-g", "--game", true, false},         {"-m", "--multiplayer", true, false},
        {"-p", "--program", true, false},      {"", "--pack-nup", true, false},
        {"", "--unpack-nup", true, false},     {"", "--edit-nup", true, false},
        {"", "--nup-name", true, false},       {"", "--nup-icon", true, false},
        {"", "--nup-version", true, false},    {"", "--nup-publisher", true, false},
    };

    std::vector<std::string> kept;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view argument = arguments[i];
        if (argument == "--") {
            break;
        }
        const auto option = std::ranges::find_if(options, [&](const Option& o) {
            return argument == o.long_name || (!o.short_name.empty() && argument == o.short_name);
        });
        if (option != std::end(options)) {
            const bool has_value = option->value && i + 1 < arguments.size();
            if (option->keep) {
                kept.emplace_back(argument);
                if (has_value) {
                    kept.push_back(arguments[i + 1]);
                }
            }
            i += has_value ? 1 : 0;
            continue;
        }
        // --name=value, or a short option with its value attached, such as -c<path>.
        const bool attached = std::ranges::any_of(options, [&](const Option& o) {
            return o.keep && o.value &&
                   (argument.starts_with(std::string{o.long_name} + "=") ||
                    (!o.short_name.empty() && argument.size() > 2 &&
                     argument.starts_with(o.short_name) && !argument.starts_with("--")));
        });
        if (attached) {
            kept.emplace_back(argument);
        }
        // Anything else is a game path, -qlaunch and the like, or a one-off option.
    }
    return kept;
}

} // namespace UpdateInstaller
