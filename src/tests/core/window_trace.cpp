// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <barrier>
#include <map>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/arm/dynarmic/window_trace.h"
#include "dynarmic/interface/A64/a64.h"

namespace {

using Core::WindowTrace::Point;
constexpr u64 Base = 0x10000000;
constexpr u64 Stop = Base + 0x800000;

class TraceEnvironment final : public Dynarmic::A64::UserCallbacks {
public:
    bool enabled{};
    Dynarmic::A64::Jit* jit{};
    std::map<u64, u32> code;
    std::map<u64, u32> data;
    std::vector<Point> points;
    std::vector<u64> values;

    std::optional<u32> MemoryReadCode(u64 address) override {
        const auto it = code.find(address);
        if (it == code.end()) {
            return std::nullopt;
        }
        return enabled ? Core::WindowTrace::InstructionFor(Base, address, it->second) : it->second;
    }

    u32 Read32(u64 address) {
        return data.at(address);
    }
    u64 MemoryRead(u64 address, size_t size) override {
        REQUIRE(size == 4);
        return Read32(address);
    }
    Dynarmic::A64::Vector MemoryRead128(u64) override {
        FAIL("Unexpected memory read");
        return {};
    }
    void MemoryWrite(u64, u64, size_t) override {
        FAIL("Unexpected memory write");
    }
    void MemoryWrite128(u64, Dynarmic::A64::Vector) override {
        FAIL("Unexpected memory write");
    }
    void CallSVC(u32) override {
        jit->HaltExecution();
    }
    void ExceptionRaised(u64 pc, Dynarmic::A64::Exception exception) override {
        REQUIRE(exception == Dynarmic::A64::Exception::Breakpoint);
        const auto point = Core::WindowTrace::Identify(Base, pc, code.at(pc));
        REQUIRE(point != Point::None);
        points.push_back(point);
        values.push_back(jit->GetRegister(0));
        Core::WindowTrace::Resume(point, *jit, *this);
    }
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 1000; }
    u64 GetCNTPCT() override { return 0; }
};

Dynarmic::A64::UserConfig Config(TraceEnvironment& env) {
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    config.enable_cycle_counting = false;
    return config;
}

} // namespace

TEST_CASE("Window diagnostics observe initialization results without changing execution",
          "[window_trace]") {
    struct Sites {
        u64 entry;
        u64 exit;
        Point entry_point;
        Point exit_point;
    };
    constexpr std::array sites{
        Sites{0x675600, 0x6757B4, Point::WindowInitialize, Point::WindowInitializeReturn},
        Sites{0x5E5980, 0x5E5AB0, Point::NativeInitialize, Point::NativeInitializeSuccess},
        Sites{0x5E5D50, 0x5E5B88, Point::Acquire, Point::NativeInitializeFailure},
        Sites{0x5E5B90, 0x6757B4, Point::Finalize, Point::WindowInitializeReturn},
    };
    for (const auto& site : sites) {
        for (const bool enabled : {false, true}) {
            for (u32 result : {0U, 1U}) {
                TraceEnvironment env;
                env.enabled = enabled;
                const u64 Entry = Base + site.entry;
                const u64 Return = Base + site.exit;
                env.code = {{Entry, 0xD503237F},
                            {Entry + 4, 0x52800000U | (result << 5)},
                            {Entry + 8, 0x14000000U |
                                            (static_cast<u32>((Return - Entry - 8) / 4) & 0x3FFFFFF)},
                            {Return, 0xD65F0FFF},
                            {Stop, 0xD4000001}};
                Dynarmic::A64::Jit jit{Config(env)};
                env.jit = &jit;
                jit.SetPC(Entry);
                jit.SetRegister(0, 123);
                for (size_t i = 1; i < 30; ++i) {
                    jit.SetRegister(i, 0x12340000 + i);
                }
                jit.SetRegister(30, Stop);
                jit.SetSP(0x20000000);
                jit.SetPstate(0xA0000000);
                jit.Run();
                REQUIRE(jit.GetRegister(0) == result);
                REQUIRE(jit.GetRegister(30) == Stop);
                REQUIRE(jit.GetSP() == 0x20000000);
                REQUIRE(jit.GetPstate() == 0xA0000000);
                for (size_t i = 1; i < 30; ++i) {
                    REQUIRE(jit.GetRegister(i) == 0x12340000 + i);
                }
                if (enabled) {
                    REQUIRE(env.points == std::vector<Point>{site.entry_point, site.exit_point});
                    REQUIRE(env.values == std::vector<u64>{123, result});
                } else {
                    REQUIRE(env.points.empty());
                }
            }
        }
    }
}

TEST_CASE("Window validation diagnostics preserve the intercepted unsigned load", "[window_trace]") {
    for (const bool enabled : {false, true}) {
        TraceEnvironment env;
        env.enabled = enabled;
        constexpr u64 Entry = Base + 0x5E6A34;
        env.code = {{Entry, 0xB9400000U | (3U << 10) | (25U << 5) | 8U},
                    {Entry + 4, 0xD4000001}};
        env.data[0x2000000C] = 0xFEDCBA98;
        Dynarmic::A64::Jit jit{Config(env)};
        env.jit = &jit;
        jit.SetPC(Entry);
        jit.SetRegister(8, ~0ULL);
        jit.SetRegister(25, 0x20000000);
        jit.SetPstate(0x60000000);
        jit.Run();
        REQUIRE(jit.GetRegister(8) == 0xFEDCBA98ULL);
        REQUIRE(jit.GetRegister(25) == 0x20000000);
        REQUIRE(jit.GetPstate() == 0x60000000);
        REQUIRE(env.points.size() == (enabled ? 1U : 0U));
    }
}

TEST_CASE("Window diagnostic probes require a registered module and matching instructions", "[window_trace]") {
    REQUIRE(Core::WindowTrace::Identify(0, 0x675600, 0xD503237F) == Point::None);
    REQUIRE(Core::WindowTrace::Identify(Base, Base - 4, 0xD503237F) == Point::None);
    REQUIRE(Core::WindowTrace::Identify(Base, Base + 0x675600, 0xD503201F) == Point::None);
    REQUIRE(Core::WindowTrace::InstructionFor(Base, Base + 0x1234, 0xD503237F) == 0xD503237F);
}

TEST_CASE("Window finalization diagnostics preserve the following unlock tail call", "[window_trace]") {
    for (const bool enabled : {false, true}) {
        TraceEnvironment env;
        env.enabled = enabled;
        constexpr u64 Entry = Base + 0x5E5D40;
        constexpr u64 Unlock = Entry + 0x40;
        env.code = {{Entry, 0xD50323FF},
                    {Entry + 4, 0x14000000U | static_cast<u32>((Unlock - Entry - 4) / 4)},
                    {Unlock, 0x52800000U | (7U << 5)},
                    {Unlock + 4, 0xD65F03C0},
                    {Stop, 0xD4000001}};
        Dynarmic::A64::Jit jit{Config(env)};
        env.jit = &jit;
        jit.SetPC(Entry);
        jit.SetRegister(0, 123);
        jit.SetRegister(30, Stop);
        jit.Run();
        REQUIRE(jit.GetRegister(0) == 7);
        REQUIRE(jit.GetRegister(30) == Stop);
        if (enabled) {
            REQUIRE(env.points == std::vector<Point>{Point::FinalizeCleared});
            REQUIRE(env.values == std::vector<u64>{123});
        } else {
            REQUIRE(env.points.empty());
        }
    }
}

TEST_CASE("Window tracing treats missing empty and zero environment values as disabled", "[window_trace]") {
    REQUIRE_FALSE(Core::WindowTrace::EnabledValue(nullptr));
    REQUIRE_FALSE(Core::WindowTrace::EnabledValue(""));
    REQUIRE_FALSE(Core::WindowTrace::EnabledValue("0"));
    REQUIRE(Core::WindowTrace::EnabledValue("1"));
}

TEST_CASE("Window diagnostics reset identities and budgets for every SDK load", "[window_trace]") {
    using Core::WindowTrace::Session;
    Core::WindowTrace::Sessions sessions;
    REQUIRE_FALSE(sessions.Find(nullptr));
    sessions.Reset(nullptr, Base);
    auto old = sessions.Find(nullptr);
    REQUIRE(old);
    REQUIRE(old->Record(Point::WindowInitialize, 17, 100, 200));
    for (u32 i = 0; i < Session::GsvLimit; ++i) {
        REQUIRE(old->TakeGsvLog());
    }
    REQUIRE_FALSE(old->TakeGsvLog());

    sessions.Reset(nullptr, Base + 0x1000000);
    auto next = sessions.Find(nullptr);
    REQUIRE(next);
    REQUIRE(next->id != old->id);
    REQUIRE(next->base == Base + 0x1000000);
    const auto identity = next->Record(Point::ValidateTexture, 17);
    REQUIRE(identity);
    REQUIRE(identity->window == 0);
    REQUIRE(identity->native == 0);
    const auto old_identity = old->Record(Point::ValidateTexture, 17);
    REQUIRE(old_identity);
    REQUIRE(old_identity->window == 100);
    REQUIRE(old_identity->native == 200);
    for (u32 i = 0; i < Session::GsvLimit; ++i) {
        REQUIRE(next->TakeGsvLog());
    }
    for (u32 i = 1; i < Session::PointLimit; ++i) {
        REQUIRE(next->Record(Point::ValidateTexture, 17));
    }
    REQUIRE_FALSE(next->TakeGsvLog());
    REQUIRE_FALSE(next->Record(Point::ValidateTexture, 17));

    sessions.Reset(nullptr, Base);
    auto fresh = sessions.Find(nullptr);
    REQUIRE(fresh->Record(Point::ValidateTexture, 17));
    REQUIRE(fresh->TakeGsvLog());
    std::weak_ptr<Session> retired = fresh;
    sessions.Reset(nullptr, 0);
    REQUIRE_FALSE(sessions.Find(nullptr));
    REQUIRE_FALSE(retired.expired());
    REQUIRE(fresh->Record(Point::ValidateTexture, 17));
    fresh.reset();
    REQUIRE(retired.expired());
}

TEST_CASE("Window diagnostic callbacks retain their session across concurrent SDK reloads", "[window_trace]") {
    using Core::WindowTrace::Session;
    Core::WindowTrace::Sessions sessions;
    sessions.Reset(nullptr, Base);
    auto old = sessions.Find(nullptr);
    std::barrier gate{9};
    std::atomic<u32> points{};
    std::atomic<u32> gsv{};
    std::atomic<bool> identities_match{true};
    std::vector<std::thread> workers;
    for (u64 i = 0; i < 8; ++i) {
        workers.emplace_back([&, old, i] {
            gate.arrive_and_wait();
            for (u32 j = 0; j < 128; ++j) {
                if (const auto identity = old->Record(Point::WindowInitialize, i, i + 1, i + 2)) {
                    ++points;
                    if (identity->window != i + 1 || identity->native != i + 2) {
                        identities_match = false;
                    }
                }
                gsv += old->TakeGsvLog();
            }
        });
    }
    sessions.Reset(nullptr, Base);
    auto current = sessions.Find(nullptr);
    gate.arrive_and_wait();
    for (auto& worker : workers) {
        worker.join();
    }
    REQUIRE(points == Session::PointLimit);
    REQUIRE(gsv == Session::GsvLimit);
    REQUIRE(identities_match);
    for (u32 i = 0; i < Session::PointLimit; ++i) {
        const auto identity = current->Record(Point::ValidateTexture, i % 8);
        REQUIRE(identity);
        REQUIRE(identity->window == 0);
        REQUIRE(identity->native == 0);
    }
    for (u32 i = 0; i < Session::GsvLimit; ++i) {
        REQUIRE(current->TakeGsvLog());
    }
    REQUIRE_FALSE(current->Record(Point::ValidateTexture, 0));
    REQUIRE_FALSE(current->TakeGsvLog());
}
