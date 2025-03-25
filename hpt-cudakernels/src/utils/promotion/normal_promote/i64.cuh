#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(i64, bool, f64);
impl_float_out_binary_promote(i64, i8, f64);
impl_float_out_binary_promote(i64, i16, f64);
impl_float_out_binary_promote(i64, i32, f64);
impl_float_out_binary_promote(i64, i64, f64);
impl_float_out_binary_promote(i64, u8, f64);
impl_float_out_binary_promote(i64, u16, f64);
impl_float_out_binary_promote(i64, u32, f64);
impl_float_out_binary_promote(i64, u64, f64);
impl_float_out_binary_promote(i64, f16, f64);
impl_float_out_binary_promote(i64, bf16, f64);
impl_float_out_binary_promote(i64, f32, f64);
impl_float_out_binary_promote(i64, f64, f64);

impl_normal_out_promote(i64, bool, i64);
impl_normal_out_promote(i64, i8, i64);
impl_normal_out_promote(i64, i16, i64);
impl_normal_out_promote(i64, i32, i64);
impl_normal_out_promote(i64, i64, i64);
impl_normal_out_promote(i64, u8, i64);
impl_normal_out_promote(i64, u16, i64);
impl_normal_out_promote(i64, u32, i64);
impl_normal_out_promote(i64, u64, i64);
impl_normal_out_promote(i64, f16, f16);
impl_normal_out_promote(i64, bf16, bf16);
impl_normal_out_promote(i64, f32, f64);
impl_normal_out_promote(i64, f64, f64);

impl_float_out_unary_promote(i64, f64);
