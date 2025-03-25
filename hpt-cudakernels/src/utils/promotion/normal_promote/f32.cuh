#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"


impl_float_out_binary_promote(f32, bool, f32);
impl_float_out_binary_promote(f32, i8, f32);
impl_float_out_binary_promote(f32, i16, f32);
impl_float_out_binary_promote(f32, i32, f32);
impl_float_out_binary_promote(f32, i64, f64);
impl_float_out_binary_promote(f32, u8, f32);
impl_float_out_binary_promote(f32, u16, f32);
impl_float_out_binary_promote(f32, u32, f32);
impl_float_out_binary_promote(f32, u64, f64);
impl_float_out_binary_promote(f32, f16, f32);
impl_float_out_binary_promote(f32, bf16, f32);
impl_float_out_binary_promote(f32, f32, f32);
impl_float_out_binary_promote(f32, f64, f64);

impl_normal_out_promote(f32, bool, f32);
impl_normal_out_promote(f32, i8, f32);
impl_normal_out_promote(f32, i16, f32);
impl_normal_out_promote(f32, i32, f32);
impl_normal_out_promote(f32, i64, f64);
impl_normal_out_promote(f32, u8, f32);
impl_normal_out_promote(f32, u16, f32);
impl_normal_out_promote(f32, u32, f32);
impl_normal_out_promote(f32, u64, f64);
impl_normal_out_promote(f32, f16, f32);
impl_normal_out_promote(f32, bf16, f32);
impl_normal_out_promote(f32, f32, f32);
impl_normal_out_promote(f32, f64, f64);

impl_float_out_unary_promote(f32, f32);