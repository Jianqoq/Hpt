#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(f16, bool, f16);
impl_float_out_binary_promote(f16, i8, f16);
impl_float_out_binary_promote(f16, i16, f16);
impl_float_out_binary_promote(f16, i32, f32);
impl_float_out_binary_promote(f16, i64, f64);
impl_float_out_binary_promote(f16, u8, f16);
impl_float_out_binary_promote(f16, u16, f16);
impl_float_out_binary_promote(f16, u32, f32);
impl_float_out_binary_promote(f16, u64, f64);
impl_float_out_binary_promote(f16, f16, f16);
impl_float_out_binary_promote(f16, bf16, f16);
impl_float_out_binary_promote(f16, f32, f32);
impl_float_out_binary_promote(f16, f64, f64);

impl_normal_out_promote(f16, bool, f16);
impl_normal_out_promote(f16, i8, f16);
impl_normal_out_promote(f16, i16, f16);
impl_normal_out_promote(f16, i32, f32);
impl_normal_out_promote(f16, i64, f64);
impl_normal_out_promote(f16, u8, f16);
impl_normal_out_promote(f16, u16, f16);
impl_normal_out_promote(f16, u32, f32);
impl_normal_out_promote(f16, u64, f64);
impl_normal_out_promote(f16, f16, f16);
impl_normal_out_promote(f16, bf16, f16);
impl_normal_out_promote(f16, f32, f32);
impl_normal_out_promote(f16, f64, f64);

impl_float_out_unary_promote(f16, f16);
