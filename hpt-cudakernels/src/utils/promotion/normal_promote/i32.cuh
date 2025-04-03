#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(i32, bool, f32, f32);
impl_float_out_binary_promote(i32, i8, f32, f32);
impl_float_out_binary_promote(i32, i16, f32, f32);
impl_float_out_binary_promote(i32, i32, f32, f32);
impl_float_out_binary_promote(i32, i64, f64, f64);
impl_float_out_binary_promote(i32, u8, f32, f32);
impl_float_out_binary_promote(i32, u16, f32, f32);
impl_float_out_binary_promote(i32, u32, f32, f32);
impl_float_out_binary_promote(i32, u64, f64, f64);
impl_float_out_binary_promote(i32, f16, f16, f32);
impl_float_out_binary_promote(i32, bf16, bf16, f32);
impl_float_out_binary_promote(i32, f32, f32, f32);
impl_float_out_binary_promote(i32, f64, f64, f64);

impl_normal_out_promote(i32, bool, i32, i32);
impl_normal_out_promote(i32, i8, i32, i32);
impl_normal_out_promote(i32, i16, i32, i32);
impl_normal_out_promote(i32, i32, i32, i32);
impl_normal_out_promote(i32, i64, i64, i64);
impl_normal_out_promote(i32, u8, i32, i32);
impl_normal_out_promote(i32, u16, i32, i32);
impl_normal_out_promote(i32, u32, i32, i32);
impl_normal_out_promote(i32, u64, i64, i64);
impl_normal_out_promote(i32, f16, f16, f32);
impl_normal_out_promote(i32, bf16, bf16, f32);
impl_normal_out_promote(i32, f32, f32, f32);
impl_normal_out_promote(i32, f64, f64, f64);

impl_float_out_unary_promote(i32, f32, f32);