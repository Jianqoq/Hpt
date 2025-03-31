#pragma once
#include "stdint.h"
#include "../type_alias.cuh"

template <typename LHS, typename RHS>
struct NormalOutPromote;

template <typename LHS, typename RHS>
struct FloatOutBinaryPromote;

template <typename IN>
struct FloatOutUnaryPromote;

#define NORMAL_PROMOTE(LHS, RHS, OUT, INTERMEDIATE) \
    template <>                                     \
    struct NormalOutPromote<LHS, RHS>               \
    {                                               \
        using Output = OUT;                         \
        using Intermediate = INTERMEDIATE;          \
    };

#define FLOAT_OUT_BINARY_PROMOTE(LHS, RHS, OUT, INTERMEDIATE) \
    template <>                                               \
    struct FloatOutBinaryPromote<LHS, RHS>                    \
    {                                                         \
        using Output = OUT;                                   \
        using Intermediate = INTERMEDIATE;                    \
    };

#define FLOAT_OUT_UNARY_PROMOTE(IN, OUT, INTERMEDIATE) \
    template <>                                        \
    struct FloatOutUnaryPromote<IN>                    \
    {                                                  \
        using Output = OUT;                            \
        using Intermediate = INTERMEDIATE;             \
    };

#define impl_float_out_binary_promote FLOAT_OUT_BINARY_PROMOTE
#define impl_normal_out_promote NORMAL_PROMOTE
#define impl_float_out_unary_promote FLOAT_OUT_UNARY_PROMOTE