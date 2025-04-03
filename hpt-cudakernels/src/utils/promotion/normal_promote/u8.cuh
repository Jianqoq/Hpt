#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "../macros.cuh"
#include "stdint.h"

impl_float_out_binary_promote(u8, bool, f16, f32);
impl_float_out_binary_promote(u8, i8, f16, f32);
impl_float_out_binary_promote(u8, i16, f16, f32);
impl_float_out_binary_promote(u8, i32, f32, f32);
impl_float_out_binary_promote(u8, i64, f64, f64);
impl_float_out_binary_promote(u8, u8, f16, f32);
impl_float_out_binary_promote(u8, u16, f16, f32);
impl_float_out_binary_promote(u8, u32, f32, f32);
impl_float_out_binary_promote(u8, u64, f64, f64);
impl_float_out_binary_promote(u8, f16, f16, f32);
impl_float_out_binary_promote(u8, bf16, bf16, f32);
impl_float_out_binary_promote(u8, f32, f32, f32);
impl_float_out_binary_promote(u8, f64, f64, f64);

impl_normal_out_promote(u8, bool, u8, u8);
impl_normal_out_promote(u8, i8, i8, i8);
impl_normal_out_promote(u8, i16, i16, i16);
impl_normal_out_promote(u8, i32, i32, i32);
impl_normal_out_promote(u8, i64, i64, i64);
impl_normal_out_promote(u8, u8, u8, u8);
impl_normal_out_promote(u8, u16, u16, u16);
impl_normal_out_promote(u8, u32, u32, u32);
impl_normal_out_promote(u8, u64, u64, u64);
impl_normal_out_promote(u8, f16, f16, f32);
impl_normal_out_promote(u8, bf16, bf16, f32);
impl_normal_out_promote(u8, f32, f32, f32);
impl_normal_out_promote(u8, f64, f64, f64);

impl_float_out_unary_promote(u8, f16, f32);
