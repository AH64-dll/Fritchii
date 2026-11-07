// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_address_wait.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Common::Nx2AddressWait {

namespace {

#ifdef _WIN32
using WaitOnAddressFn = BOOL(WINAPI*)(volatile VOID*, PVOID, SIZE_T, DWORD);
using WakeByAddressFn = VOID(WINAPI*)(PVOID);

struct AddressApi {
    WaitOnAddressFn wait{};
    WakeByAddressFn wake_one{};
    WakeByAddressFn wake_all{};
};

const AddressApi& Api() {
    static const AddressApi api = [] {
        AddressApi result{};
        HMODULE module = GetModuleHandleA("KernelBase.dll");
        if (module == nullptr) {
            module = LoadLibraryA("api-ms-win-core-synch-l1-2-0.dll");
        }
        if (module == nullptr) {
            return result;
        }
        result.wait = reinterpret_cast<WaitOnAddressFn>(GetProcAddress(module, "WaitOnAddress"));
        result.wake_one =
            reinterpret_cast<WakeByAddressFn>(GetProcAddress(module, "WakeByAddressSingle"));
        result.wake_all =
            reinterpret_cast<WakeByAddressFn>(GetProcAddress(module, "WakeByAddressAll"));
        if (result.wait == nullptr || result.wake_one == nullptr || result.wake_all == nullptr) {
            return AddressApi{};
        }
        return result;
    }();
    return api;
}
#endif

} // Anonymous namespace

void Wait(std::atomic<u32>& word, u32 expected) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wait != nullptr) {
        api.wait(static_cast<void*>(&word), &expected, sizeof(u32), INFINITE);
        return;
    }
#endif
    word.wait(expected, std::memory_order_relaxed);
}

void WakeOne(std::atomic<u32>& word) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wake_one != nullptr) {
        api.wake_one(static_cast<void*>(&word));
        return;
    }
#endif
    word.notify_one();
}

void WakeAll(std::atomic<u32>& word) {
#ifdef _WIN32
    if (const AddressApi& api = Api(); api.wake_all != nullptr) {
        api.wake_all(static_cast<void*>(&word));
        return;
    }
#endif
    word.notify_all();
}

} // namespace Common::Nx2AddressWait
