#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"


impl_float_out_binary_promote(f64, bool, f64);
impl_float_out_binary_promote(f64, i8, f64);
impl_float_out_binary_promote(f64, i16, f64);
impl_float_out_binary_promote(f64, i32, f64);
impl_float_out_binary_promote(f64, i64, f64);
impl_float_out_binary_promote(f64, u8, f64);
impl_float_out_binary_promote(f64, u16, f64);
impl_float_out_binary_promote(f64, u32, f64);
impl_float_out_binary_promote(f64, u64, f64);
impl_float_out_binary_promote(f64, f16, f64);
impl_float_out_binary_promote(f64, bf16, f64);
impl_float_out_binary_promote(f64, f32, f64);
impl_float_out_binary_promote(f64, f64, f64);

impl_normal_out_promote(f64, bool, f64);
impl_normal_out_promote(f64, i8, f64);
impl_normal_out_promote(f64, i16, f64);
impl_normal_out_promote(f64, i32, f64);
impl_normal_out_promote(f64, i64, f64);
impl_normal_out_promote(f64, u8, f64);
impl_normal_out_promote(f64, u16, f64);
impl_normal_out_promote(f64, u32, f64);
impl_normal_out_promote(f64, u64, f64);
impl_normal_out_promote(f64, f16, f64);
impl_normal_out_promote(f64, bf16, f64);
impl_normal_out_promote(f64, f32, f64);
impl_normal_out_promote(f64, f64, f64);

impl_float_out_unary_promote(f64, f64);
