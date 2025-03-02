#pragma once

#define CHECK_FLOAT_TYPE(T) \
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be half, __nv_bfloat16, float, or double");
