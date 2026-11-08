// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <catch2/catch_test_macros.hpp>

#include "hid_core/resources/six_axis/sensor_info.h"

using Core::HID::SixAxisSensorIcInformation;
using Service::HID::DefaultSixAxisSensorIcInformation;

TEST_CASE("Six axis defaults provide positive sensor fusion ranges", "[hid][sensor_info]") {
    constexpr auto info = DefaultSixAxisSensorIcInformation();
    // The guest SDK aborts when either physical range is zero or negative.
    REQUIRE(info.acceleration_range == 8.0f);
    REQUIRE(info.angular_rate == 2000.0f);
    REQUIRE(info.acceleration_range * 9.80665f > 0.0f);
    REQUIRE(info.angular_rate / 360.0f > 0.0f);
    for (const float value : std::bit_cast<std::array<float, 50>>(info)) {
        REQUIRE(std::isfinite(value));
    }
}

TEST_CASE("Six axis defaults preserve calibration bounds", "[hid][sensor_info]") {
    constexpr auto info = DefaultSixAxisSensorIcInformation();
    for (std::size_t axis = 0; axis < 3; ++axis) {
        REQUIRE(info.unknown_gyro_data1[axis] == -10.0f);
        REQUIRE(info.unknown_gyro_data1[axis + 3] == 10.0f);
        REQUIRE(info.unknown_accel_data1[axis] == -0.0612f);
        REQUIRE(info.unknown_accel_data1[axis + 3] == 0.0612f);
    }
    for (std::size_t i = 0; i < 9; ++i) {
        const bool diagonal = i % 4 == 0;
        REQUIRE(info.unknown_gyro_data2[i] == (diagonal ? 0.95f : -0.003f));
        REQUIRE(info.unknown_gyro_data3[i] == (diagonal ? 1.05f : 0.003f));
        REQUIRE(info.unknown_accel_data2[i] == info.unknown_gyro_data2[i]);
        REQUIRE(info.unknown_accel_data3[i] == info.unknown_gyro_data3[i]);
    }
}

TEST_CASE("Six axis IPC payload remains trivial with unchanged layout", "[hid][sensor_info]") {
    STATIC_REQUIRE(std::is_trivial_v<SixAxisSensorIcInformation>);
    STATIC_REQUIRE(sizeof(SixAxisSensorIcInformation) == 0xc8);
    STATIC_REQUIRE(offsetof(SixAxisSensorIcInformation, angular_rate) == 0);
    STATIC_REQUIRE(offsetof(SixAxisSensorIcInformation, acceleration_range) == 0x64);
    [[maybe_unused]] auto left = DefaultSixAxisSensorIcInformation();
    const auto right = DefaultSixAxisSensorIcInformation();
    left.angular_rate = 1000.0f;
    REQUIRE(right.angular_rate == 2000.0f);
}
