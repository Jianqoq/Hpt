#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"


impl_float_out_binary_promote(i16, bool, f16);
impl_float_out_binary_promote(i16, i8, f16);
impl_float_out_binary_promote(i16, i16, f16);
impl_float_out_binary_promote(i16, i32, f32);
impl_float_out_binary_promote(i16, i64, f64);
impl_float_out_binary_promote(i16, u8, f16);
impl_float_out_binary_promote(i16, u16, f16);
impl_float_out_binary_promote(i16, u32, f32);
impl_float_out_binary_promote(i16, u64, f64);
impl_float_out_binary_promote(i16, f16, f16);
impl_float_out_binary_promote(i16, bf16, bf16);
impl_float_out_binary_promote(i16, f32, f32);
impl_float_out_binary_promote(i16, f64, f64);

impl_normal_out_promote(i16, bool, i16);
impl_normal_out_promote(i16, i8, i16);
impl_normal_out_promote(i16, i16, i16);
impl_normal_out_promote(i16, i32, i32);
impl_normal_out_promote(i16, i64, i64);
impl_normal_out_promote(i16, u8, i16);
impl_normal_out_promote(i16, u16, i16);
impl_normal_out_promote(i16, u32, i32);
impl_normal_out_promote(i16, u64, i64);
impl_normal_out_promote(i16, f16, f16);
impl_normal_out_promote(i16, bf16, bf16);
impl_normal_out_promote(i16, f32, f32);
impl_normal_out_promote(i16, f64, f64);

impl_float_out_unary_promote(i16, f16);
