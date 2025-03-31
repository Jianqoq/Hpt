#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(i8, bool, f16, f16);
impl_float_out_binary_promote(i8, i8, f16, f16);
impl_float_out_binary_promote(i8, i16, f16, f16);
impl_float_out_binary_promote(i8, i32, f32, f32);
impl_float_out_binary_promote(i8, i64, f64, f64);
impl_float_out_binary_promote(i8, u8, f16, f16);
impl_float_out_binary_promote(i8, u16, f16, f16);
impl_float_out_binary_promote(i8, u32, f32, f32);
impl_float_out_binary_promote(i8, u64, f64, f64);
impl_float_out_binary_promote(i8, f16, f16, f16);
impl_float_out_binary_promote(i8, bf16, bf16, f16);
impl_float_out_binary_promote(i8, f32, f32, f32);
impl_float_out_binary_promote(i8, f64, f64, f64);

impl_normal_out_promote(i8, bool, i8, i8);
impl_normal_out_promote(i8, i8, i8, i8);
impl_normal_out_promote(i8, i16, i16, i16);
impl_normal_out_promote(i8, i32, i32, i32);
impl_normal_out_promote(i8, i64, i64, i64);
impl_normal_out_promote(i8, u8, i8, i8);
impl_normal_out_promote(i8, u16, i16, i16);
impl_normal_out_promote(i8, u32, i32, i32);
impl_normal_out_promote(i8, u64, i64, i64);
impl_normal_out_promote(i8, f16, f16, f32);
impl_normal_out_promote(i8, bf16, bf16, f32);
impl_normal_out_promote(i8, f32, f32, f32);
impl_normal_out_promote(i8, f64, f64, f64);

impl_float_out_unary_promote(i8, f16, f32);
