#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(bf16, bool, bf16, f32);
impl_float_out_binary_promote(bf16, i8, bf16, f32);
impl_float_out_binary_promote(bf16, i16, bf16, f32);
impl_float_out_binary_promote(bf16, i32, f32, f32);
impl_float_out_binary_promote(bf16, i64, f64, f64);
impl_float_out_binary_promote(bf16, u8, bf16, f32);
impl_float_out_binary_promote(bf16, u16, bf16, f32);
impl_float_out_binary_promote(bf16, u32, f32, f32);
impl_float_out_binary_promote(bf16, u64, f64, f64);
impl_float_out_binary_promote(bf16, bf16, bf16, f32);
impl_float_out_binary_promote(bf16, f16, bf16, f32);
impl_float_out_binary_promote(bf16, f32, f32, f32);
impl_float_out_binary_promote(bf16, f64, f64, f64);

impl_normal_out_promote(bf16, bool, bf16, f32);
impl_normal_out_promote(bf16, i8, bf16, f32);
impl_normal_out_promote(bf16, i16, bf16, f32);
impl_normal_out_promote(bf16, i32, f32, f32);
impl_normal_out_promote(bf16, i64, f64, f64);
impl_normal_out_promote(bf16, u8, bf16, f32);
impl_normal_out_promote(bf16, u16, bf16, f32);
impl_normal_out_promote(bf16, u32, f32, f32);
impl_normal_out_promote(bf16, u64, f64, f64);
impl_normal_out_promote(bf16, bf16, bf16, f32);
impl_normal_out_promote(bf16, f16, bf16, f32);
impl_normal_out_promote(bf16, f32, f32, f32);
impl_normal_out_promote(bf16, f64, f64, f64);

impl_float_out_unary_promote(bf16, bf16, f32);