#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(bool, bool, f16);
impl_float_out_binary_promote(bool, i8, f16);
impl_float_out_binary_promote(bool, i16, f16);
impl_float_out_binary_promote(bool, i32, f32);
impl_float_out_binary_promote(bool, i64, f64);
impl_float_out_binary_promote(bool, u8, f16);
impl_float_out_binary_promote(bool, u16, f16);
impl_float_out_binary_promote(bool, u32, f32);
impl_float_out_binary_promote(bool, u64, f64);
impl_float_out_binary_promote(bool, f16, f16);
impl_float_out_binary_promote(bool, bf16, bf16);
impl_float_out_binary_promote(bool, f32, f32);
impl_float_out_binary_promote(bool, f64, f64);

impl_normal_out_promote(bool, bool, bool);
impl_normal_out_promote(bool, i8, i8);
impl_normal_out_promote(bool, i16, i16);
impl_normal_out_promote(bool, i32, i32);
impl_normal_out_promote(bool, i64, i64);
impl_normal_out_promote(bool, u8, u8);
impl_normal_out_promote(bool, u16, u16);
impl_normal_out_promote(bool, u32, u32);
impl_normal_out_promote(bool, u64, u64);
impl_normal_out_promote(bool, f16, f16);
impl_normal_out_promote(bool, bf16, bf16);
impl_normal_out_promote(bool, f32, f32);
impl_normal_out_promote(bool, f64, f64);

impl_float_out_unary_promote(bool, f16);