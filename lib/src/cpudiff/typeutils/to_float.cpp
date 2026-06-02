//
// Created by iliya on 5/20/26.
//

#include <cstring>
#include <cstdint>
#include <algorithm>
#include <stdfloat>

#include "to_float.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"

void dtype_to_float(float *dst, const void *src, size_t num_elements, Dtype dtype) {
    switch (dtype) {
        case Dtype::F32: {
            if constexpr (FloatDtype == Dtype::F32) {
                std::memcpy(dst, src, num_elements * sizeof(float));
            } else {
                const auto *s = static_cast<const std::float32_t*>(src);
                for (size_t i = 0; i < num_elements; ++i) {
                    dst[i] = static_cast<float>(s[i]);
                }
            }
        }
        case Dtype::F16: {
            if constexpr (FloatDtype == Dtype::F16) {
                std::memcpy(dst, src, num_elements * sizeof(float));
            } else {
                const auto *s = static_cast<const std::float16_t*>(src);
                for (size_t i = 0; i < num_elements; ++i) {
                    dst[i] = static_cast<float>(s[i]);
                }
            }
            break;
        }
        case Dtype::BF16: {
            const auto *s = static_cast<const std::bfloat16_t*>(src);
            for (size_t i = 0; i < num_elements; ++i) {
                dst[i] = static_cast<float>(s[i]);
            }
            break;
        }
        case Dtype::I32: {
            const auto *s = static_cast<const int32_t*>(src);
            for (size_t i = 0; i < num_elements; ++i) {
                dst[i] = static_cast<float>(s[i]);
            }
            break;
        }
        case Dtype::I64: {
            const auto *s = static_cast<const int64_t*>(src);
            for (size_t i = 0; i < num_elements; ++i) {
                dst[i] = static_cast<float>(s[i]);
            }
            break;
        }
        default:
            throw std::logic_error("This format is unsupported at now");
    }
}


#pragma clang diagnostic pop