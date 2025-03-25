#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(u64, bool, f64);
impl_float_out_binary_promote(u64, i8, f64);
impl_float_out_binary_promote(u64, i16, f64);
impl_float_out_binary_promote(u64, i32, f64);
impl_float_out_binary_promote(u64, i64, f64);
impl_float_out_binary_promote(u64, u8, f64);
impl_float_out_binary_promote(u64, u16, f64);
impl_float_out_binary_promote(u64, u32, f64);
impl_float_out_binary_promote(u64, u64, f64);
impl_float_out_binary_promote(u64, f16, f64);
impl_float_out_binary_promote(u64, bf16, bf16);
impl_float_out_binary_promote(u64, f32, f64);
impl_float_out_binary_promote(u64, f64, f64);

impl_normal_out_promote(u64, bool, u64);
impl_normal_out_promote(u64, i8, i64);
impl_normal_out_promote(u64, i16, i64);
impl_normal_out_promote(u64, i32, i64);
impl_normal_out_promote(u64, i64, i64);
impl_normal_out_promote(u64, u8, u64);
impl_normal_out_promote(u64, u16, u64);
impl_normal_out_promote(u64, u32, u64);
impl_normal_out_promote(u64, u64, u64);
impl_normal_out_promote(u64, f16, f16);
impl_normal_out_promote(u64, bf16, bf16);
impl_normal_out_promote(u64, f32, f32);
impl_normal_out_promote(u64, f64, f64);

impl_float_out_unary_promote(u64, f64);
