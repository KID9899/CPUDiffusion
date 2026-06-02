//
// Created by iliya on 5/20/26.
//

#include <string>
#include <stdexcept>
#include <limits>

#pragma once

#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"

enum class Dtype { F32, F16, BF16, I32, I64, UNKNOWN };

constexpr Dtype FloatDtype = (
    (sizeof(float) == 4 && std::numeric_limits<float>::is_iec559)
    ? Dtype::F32
    : (
        (sizeof(float) == 2 && std::numeric_limits<float>::is_iec559)
        ? Dtype::F16
        : Dtype::UNKNOWN
    )
);
typedef std::conditional<
        (sizeof(float) == 4 && std::numeric_limits<float>::is_iec559),
        int32_t,
        std::conditional<
                (sizeof(float) == 2 && std::numeric_limits<float>::is_iec559),
                int16_t,
                void
        >::type
    >::type CompatibleInt;
constexpr int32_t COMPATIBLE_INT_MAX = (int32_t{1} << std::numeric_limits<float>::digits) - int32_t{1};

inline Dtype dtype_from_string(const std::string &s) {
    if (s == "F32") return Dtype::F32;
    if (s == "F16") return Dtype::F16;
    if (s == "BF16") return Dtype::BF16;
    if (s == "I32") return Dtype::I32;
    if (s == "I64") return Dtype::I64;
    throw std::runtime_error("Unknown dtype: " + s);
}

#pragma clang diagnostic pop
