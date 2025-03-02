/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cstdint>
#include <cmath>
#include <type_traits>

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ void divmod(T a, T b, T &div, T &mod)
{
    div = a / b;
    mod = a % b;
}

/**
 * log2 computation, what's the
 * difference between the below codes and
 * log2_up/down codes?
 */
template <typename value_t>
value_t __device__ clz(value_t x)
{
    for (int i = 31; i >= 0; --i)
    {
        if ((1 << i) & x)
            return value_t(31 - i);
    }
    return value_t(32);
}

template <typename value_t>
value_t __device__ find_log2(value_t x)
{
    int a = int(31 - clz(x));
    a += (x & (x - 1)) != 0; // Round up, add 1 if not a power of 2.
    return a;
}

/**
 * Find divisor, using find_log2
 */
__device__ void find_divisor(unsigned int &mul, unsigned int &shr, unsigned int denom)
{
    if (denom == 1)
    {
        mul = 0;
        shr = 0;
    }
    else
    {
        unsigned int p = 31 + find_log2(denom);
        unsigned m = unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

        mul = m;
        shr = p - 32;
    }
}

/**
 * Find quotient and remainder using device-side intrinsics
 */
__device__ void fast_divmod(int &quo, int &rem, int src, int div, unsigned int mul, unsigned int shr)
{

#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = int((div != 1) ? int(((int64_t)src * mul) >> 32) >> shr : src);
#endif

    // The remainder.
    rem = src - (quo * div);
}

// For long int input
__device__ void fast_divmod(int &quo, int64_t &rem, int64_t src, int div, unsigned int mul, unsigned int shr)
{

#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = int((div != 1) ? ((src * mul) >> 32) >> shr : src);
#endif
    // The remainder.
    rem = src - (quo * div);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod
{
    using value_div_type = int;
    using value_mod_type = int64_t;
    int32_t divisor = 1;
    uint32_t multiplier = 0u;
    uint32_t shift_right = 0u;

    // Find quotient and remainder using device-side intrinsics
    __device__ void fast_divmod(int &quotient, int &remainder, int dividend) const
    {

#if defined(__CUDA_ARCH__)
        // Use IMUL.HI if divisor != 1, else simply copy the source.
        quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
        quotient = int((divisor != 1) ? int(((int64_t)dividend * multiplier) >> 32) >> shift_right : dividend);
#endif

        // The remainder.
        remainder = dividend - (quotient * divisor);
    }

    /// For long int input
    __device__ void fast_divmod(int &quotient, int64_t &remainder, int64_t dividend) const
    {

#if defined(__CUDA_ARCH__)
        // Use IMUL.HI if divisor != 1, else simply copy the source.
        quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
        quotient = int((divisor != 1) ? ((dividend * multiplier) >> 32) >> shift_right : dividend);
#endif
        // The remainder.
        remainder = dividend - (quotient * divisor);
    }

    /// Construct the FastDivmod object, in host code ideally.
    ///
    /// This precomputes some values based on the divisor and is computationally expensive.

    FastDivmod() = default;

    __device__
    FastDivmod(int divisor_) : divisor(divisor_)
    {
        if (divisor != 1)
        {
            unsigned int p = 31 + find_log2(divisor);
            unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

            multiplier = m;
            shift_right = p - 32;
        }
    }

    /// Computes integer division and modulus using precomputed values. This is computationally
    /// inexpensive.
    __device__ void operator()(int &quotient, int &remainder, int dividend) const
    {
        fast_divmod(quotient, remainder, dividend);
    }

    /// Computes integer division using precomputed values. This is computationally
    /// inexpensive.
    __device__ int div(int dividend) const
    {
        int quotient, remainder;
        fast_divmod(quotient, remainder, dividend);
        return quotient;
    }

    /// Alias for `div` to match the interface of FastDivmodU64
    __device__ int divide(int dividend) const
    {
        return div(dividend);
    }

    /// Computes integer division and modulus using precomputed values. This is computationally
    /// inexpensive.
    ///
    /// Simply returns the quotient
    __device__ int divmod(int &remainder, int dividend) const
    {
        int quotient;
        fast_divmod(quotient, remainder, dividend);
        return quotient;
    }

    /// Computes integer division and modulus using precomputed values. This is computationally
    /// inexpensive.
    __device__ void operator()(int &quotient, int64_t &remainder, int64_t dividend) const
    {
        fast_divmod(quotient, remainder, dividend);
    }

    /// Computes integer division and modulus using precomputed values. This is computationally
    /// inexpensive.
    __device__ int divmod(int64_t &remainder, int64_t dividend) const
    {
        int quotient;
        fast_divmod(quotient, remainder, dividend);
        return quotient;
    }

    /// Returns the divisor when cast to integer
    __device__
    operator int() const { return divisor; }
};
/////////////////////////////////////////////////////////////////////////////////////////////////
