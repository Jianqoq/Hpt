use crate::{
    ops::tensor::reduce::kernels::{
        contiguous_reduce_dim_include, contiguous_reduce_dim_include_simd, fast_reduce_no_simd,
        fast_reduce_simd, reduce_dim_not_include, reduce_dim_not_include_simd, uncontiguous_reduce_dim_include, uncontiguous_reduce_dim_not_include,
    }, Tensor
};
use hpt_common::{error::base::TensorError, shape::shape_utils::{mt_intervals, mt_intervals_simd}};
use hpt_iterator::TensorIterator;
use hpt_traits::tensor::TensorInfo;
use hpt_types::{into_vec::IntoVec, promote_float_unary, promote_normal_unary};
use hpt_types::{
    dtype::DType,
    dtype::ToDType,
    into_scalar::Cast,
    traits::SimdSelect,
    type_promote::{Eval, Eval2, NormalOut},
};
use num_complex::ComplexFloat;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use hpt_iterator::iterator_traits::StridedIterator;
use super::{reduce_template::{contiguous_reduce_template, uncontiguos_reduce_template}, reduce_utils::{ReductionPreprocessor, UCReductionPreprocessor}};
use half::{bf16, f16};
use hpt_types::dtype::TypeCommon;
use hpt_types::traits::VecTrait;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::dtype::FloatConst;
use hpt_types::type_promote::FloatOutBinary;
use hpt_common::axis::axis::process_axes;
use hpt_types::promote_eval;

#[duplicate::duplicate_item(
    func_name                   in_type   trait_name        pre_op                                                  pre_op_no_cast                  cumulate_op                 post_op                                     vec_pre_op                                                                                              vec_preop_no_cast                                                       vec_cumulate                            vec_post_op;
    [contiguous_sum_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_prod_f32]       [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_prod_f16]       [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_prod_bf16]      [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_prod_i8]        [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_prod_u8]        [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_min_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._min(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_min_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._min(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_min_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._min(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_min_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._min(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_min_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._min(y)]      [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_max_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._max(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_max_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._max(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_max_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._max(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_max_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._max(y)]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_max_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x._max(y)]      [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_reducel1_f32]   [f32]     [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x._abs()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];  
    [contiguous_reducel1_f16]   [f16]     [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x._abs()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel1_bf16]  [bf16]    [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x._abs()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel1_i8]    [i8]      [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x._abs()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel1_u8]    [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_sumsquare_f32]  [f32]     [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sumsquare_f16]  [f16]     [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sumsquare_bf16] [bf16]    [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sumsquare_i8]   [i8]      [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_sumsquare_u8]   [u8]      [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_all_f32]        [f32]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x & y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_all_f16]        [f16]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x & y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_all_bf16]       [bf16]    [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x & y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_all_i8]         [i8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x & y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_all_u8]         [u8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x & y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_any_f32]        [f32]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x | y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_any_f16]        [f16]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x | y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_any_bf16]       [bf16]    [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x | y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_any_i8]         [i8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x | y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_any_u8]         [u8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>]                        [|x: InVec| x.into_vec()]                                                                               [|x: OutVec| x]                                                         [|x: OutVec, y: OutVec| x | y]          [None::<fn(OutVec) -> OutVec>];
    
    [contiguous_nansum_f32]     [f32]     [NormalOut]       [|x: T| if x.is_nan() {0.0} else {x}]                   [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ZERO), x)]                             [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nansum_f16]     [f16]     [NormalOut]       [|x: T| if x.is_nan() {f16::ZERO} else {x}]             [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ZERO), x)]                             [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nansum_bf16]    [bf16]    [NormalOut]       [|x: T| if x.is_nan() {bf16::ZERO} else {x}]            [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ZERO), x)]                             [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nansum_i8]      [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nansum_u8]      [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];

    [contiguous_nanprod_f32]    [f32]     [NormalOut]       [|x: T| if x.is_nan() {1.0} else {x}]                   [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ONE), x)]                              [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nanprod_f16]    [f16]     [NormalOut]       [|x: T| if x.is_nan() {f16::ONE} else {x}]              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ONE), x)]                              [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nanprod_bf16]   [bf16]    [NormalOut]       [|x: T| if x.is_nan() {bf16::ONE} else {x}]             [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x.__is_nan().select(<T as TypeCommon>::Vec::splat(T::ONE), x)]                              [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nanprod_i8]     [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_nanprod_u8]     [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>]                        [|x: InVec| x]                                                                                          [vec_preop]                                                             [|x: OutVec, y: OutVec| x * y]          [None::<fn(OutVec) -> OutVec>];

    [contiguous_reducel2_f32]   [f32]     [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel2_f16]   [f16]     [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel2_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| x * x]                                                                                      [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel2_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x * x}]                  [|x: O| x * x]                  [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| {let x: OutVec = x.into_vec(); x * x}]                                                      [|x: OutVec| x * x]                                                     [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];
    [contiguous_reducel2_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x * x}]                  [|x: O| x * x]                  [|a: O, b: O| a + b]        [None::<fn(O) -> O>]                        [|x: InVec| {let x: OutVec = x.into_vec(); x * x}]                                                      [|x: OutVec| x * x]                                                     [|x: OutVec, y: OutVec| x + y]          [None::<fn(OutVec) -> OutVec>];

    [contiguous_reducel3_f32]   [f32]     [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))]      [|x: InVec| x._abs()._pow(<T as TypeCommon>::Vec::splat(T::THREE))]                                     [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._pow(<O as TypeCommon>::Vec::splat(O::ONE/O::THREE)))];
    [contiguous_reducel3_f16]   [f16]     [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))]      [|x: InVec| x._abs()._pow(<T as TypeCommon>::Vec::splat(T::THREE))]                                     [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._pow(<O as TypeCommon>::Vec::splat(O::ONE/O::THREE)))];
    [contiguous_reducel3_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))]      [|x: InVec| x._abs()._pow(<T as TypeCommon>::Vec::splat(T::THREE))]                                     [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._pow(<O as TypeCommon>::Vec::splat(O::ONE/O::THREE)))];
    [contiguous_reducel3_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.abs().powi(3)}]        [|x: O| x.abs().powi(3)]        [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))]      [|x: InVec| {let x: OutVec = x.into_vec(); x._abs()._pow(<O as TypeCommon>::Vec::splat(O::THREE))}]     [|x: OutVec| x._abs()._pow(<O as TypeCommon>::Vec::splat(O::THREE))]    [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._pow(<O as TypeCommon>::Vec::splat(O::ONE/O::THREE)))];
    [contiguous_reducel3_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.abs().powi(3)}]        [|x: O| x.abs().powi(3)]        [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))]      [|x: InVec| {let x: OutVec = x.into_vec(); x._abs()._pow(<O as TypeCommon>::Vec::splat(O::THREE))}]     [|x: OutVec| x._abs()._pow(<O as TypeCommon>::Vec::splat(O::THREE))]    [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._pow(<O as TypeCommon>::Vec::splat(O::ONE/O::THREE)))];

    [contiguous_logsumexp_f32]   [f32]     [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())]                       [|x: InVec| x._exp()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._ln())];
    [contiguous_logsumexp_f16]   [f16]     [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())]                       [|x: InVec| x._exp()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._ln())];
    [contiguous_logsumexp_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())]                       [|x: InVec| x._exp()]                                                                                   [vec_preop]                                                             [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._ln())];
    [contiguous_logsumexp_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.exp()}]               [|x: O| x.exp()]                [|a: O, b: O| a + b]        [Some(|x: O| x.ln())]                       [|x: InVec| {let x: OutVec = x.into_vec(); x._exp()}]                                                   [|x: OutVec| x._exp()]                                                  [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._ln())];
    [contiguous_logsumexp_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.exp()}]               [|x: O| x.exp()]                [|a: O, b: O| a + b]        [Some(|x: O| x.ln())]                       [|x: InVec| {let x: OutVec = x.into_vec(); x._exp()}]                                                   [|x: OutVec| x._exp()]                                                  [|x: OutVec, y: OutVec| x + y]          [Some(|x: OutVec| x._ln())];
)]
pub(crate) fn func_name(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
) -> std::result::Result<Tensor, TensorError> {
    type T = in_type;
    type O = <T as trait_name>::Output;
    type InVec = <T as TypeCommon>::Vec;
    type OutVec = <O as TypeCommon>::Vec;
    assert_eq!(res_dtype, O::to_dtype());
    let preop = pre_op;
    let cumulate = cumulate_op;
    let postop = post_op;
    let preop_no_cast = pre_op_no_cast;
    let vec_preop = vec_pre_op;
    let vec_postop = vec_post_op;
    let res_shape = a.layout.reduce(axes, keepdims)?.shape().clone();
    let res = contiguous_reduce_template(
        &a,
        axes,
        init_val,
        keepdims,
        init_out,
        res_dtype,
        c,
        |res| {
            let res = res as *mut O;
            let ptr = a.ptr();
            let raw =
                unsafe { std::slice::from_raw_parts_mut(ptr.ptr as *mut T, a.size() as usize) };
            let init_val: O = init_val.cast();
            let val = raw
                .par_iter()
                .fold(|| init_val, |acc, &x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                unsafe { *res = postop(val) };
            } else {
                unsafe { *res = val };
            }
        },
        |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let iterators = ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
            );
            iterators.into_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                if T::STR == O::STR {
                    contiguous_reduce_dim_include_simd(
                        init_val.cast(),
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr.cast::<O>(),
                        result_ptr_c.cast::<O>(),
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop_no_cast,
                        cumulate,
                        vec_preop_no_cast,
                        vec_cumulate,
                        postop,
                    );
                } else {
                    contiguous_reduce_dim_include(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
        |num_threads, inner_loop_size, result, a| {
            let intervals = mt_intervals_simd(inner_loop_size, num_threads, OutVec::SIZE);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); a.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[a.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, res)| {
                    let inp_ptr = inp.ptr::<T>();
                    let res_ptr = res.ptr::<O>();

                    let inner_loop_size = *res.shape().last().unwrap() as isize;
                    let outer_loop_size = (inp.size() as isize) / inner_loop_size;
                    if O::BYTE_SIZE == T::BYTE_SIZE {
                        fast_reduce_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            <O as TypeCommon>::Vec::SIZE as isize,
                            preop,
                            cumulate,
                            postop,
                            vec_preop,
                            vec_cumulate,
                            vec_postop,
                        );
                    } else {
                        fast_reduce_no_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            preop,
                            cumulate,
                            postop,
                        );
                    }
                });
        },
        |num_threads,
         outer_loop_size,
         inner_loop_size,
         inner_loop_size_2,
         result,
         transposed_tensor| {
            let iterators = ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
            );
            iterators.into_par_iter().for_each(|iterator| {
                let result_ptr_c = iterator.res_ptrs.cast::<O>();
                let a_data_ptr = iterator.ptrs.cast::<T>();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                let inp_strides = &iterator.strides;
                let inp_shape = &iterator.a_shape;
                let mut prg1 = iterator.prg.clone();
                let mut prg2 = iterator.a_prg.clone();
                if O::BYTE_SIZE == T::BYTE_SIZE {
                    reduce_dim_not_include_simd(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &inp_strides,
                        &inp_shape,
                        &mut prg1,
                        &mut prg2,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                        vec_preop,
                        vec_cumulate,
                        vec_postop,
                    );
                } else {
                    reduce_dim_not_include(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &inp_strides,
                        &inp_shape,
                        &mut prg1,
                        &mut prg2,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
    )?;
    res.reshape(&res_shape)
}

#[duplicate::duplicate_item(
    func_name                     in_type   trait_name        pre_op                                                  pre_op_no_cast                  cumulate_op                 post_op;
    [uncontiguous_sum_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sum_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sum_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sum_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sum_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    
    [uncontiguous_prod_f32]       [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_prod_f16]       [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_prod_bf16]      [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_prod_i8]        [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_prod_u8]        [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    
    [uncontiguous_min_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>];
    [uncontiguous_min_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>];
    [uncontiguous_min_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>];
    [uncontiguous_min_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>];
    [uncontiguous_min_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._min(b)]    [None::<fn(O) -> O>];
    
    [uncontiguous_max_f32]        [f32]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>];
    [uncontiguous_max_f16]        [f16]     [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>];
    [uncontiguous_max_bf16]       [bf16]    [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>];
    [uncontiguous_max_i8]         [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>];
    [uncontiguous_max_u8]         [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a._max(b)]    [None::<fn(O) -> O>];
    
    [uncontiguous_reducel1_f32]   [f32]     [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];  
    [uncontiguous_reducel1_f16]   [f16]     [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel1_bf16]  [bf16]    [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel1_i8]    [i8]      [NormalOut]       [|x: T| x.abs()]                                        [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel1_u8]    [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    
    [uncontiguous_sumsquare_f32]  [f32]     [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sumsquare_f16]  [f16]     [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sumsquare_bf16] [bf16]    [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sumsquare_i8]   [i8]      [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_sumsquare_u8]   [u8]      [NormalOut]       [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    
    [uncontiguous_all_f32]        [f32]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>];
    [uncontiguous_all_f16]        [f16]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>];
    [uncontiguous_all_bf16]       [bf16]    [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>];
    [uncontiguous_all_i8]         [i8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>];
    [uncontiguous_all_u8]         [u8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a & b]        [None::<fn(O) -> O>];
    
    [uncontiguous_any_f32]        [f32]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>];
    [uncontiguous_any_f16]        [f16]     [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>];
    [uncontiguous_any_bf16]       [bf16]    [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>];
    [uncontiguous_any_i8]         [i8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>];
    [uncontiguous_any_u8]         [u8]      [Eval]            [|x: T| x._is_true()]                                   [|x: O| x]                      [|a: O, b: O| a | b]        [None::<fn(O) -> O>];
    
    [uncontiguous_nansum_f32]     [f32]     [NormalOut]       [|x: T| if x.is_nan() {0.0} else {x}]                   [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_nansum_f16]     [f16]     [NormalOut]       [|x: T| if x.is_nan() {f16::ZERO} else {x}]             [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_nansum_bf16]    [bf16]    [NormalOut]       [|x: T| if x.is_nan() {bf16::ZERO} else {x}]            [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_nansum_i8]      [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_nansum_u8]      [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];

    [uncontiguous_nanprod_f32]    [f32]     [NormalOut]       [|x: T| if x.is_nan() {1.0} else {x}]                   [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_nanprod_f16]    [f16]     [NormalOut]       [|x: T| if x.is_nan() {f16::ONE} else {x}]              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_nanprod_bf16]   [bf16]    [NormalOut]       [|x: T| if x.is_nan() {bf16::ONE} else {x}]             [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_nanprod_i8]     [i8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];
    [uncontiguous_nanprod_u8]     [u8]      [NormalOut]       [|x: T| x]                                              [preop]                         [|a: O, b: O| a * b]        [None::<fn(O) -> O>];

    [uncontiguous_reducel2_f32]   [f32]     [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel2_f16]   [f16]     [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel2_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x * x]                                          [preop]                         [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel2_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x * x}]                  [|x: O| x * x]                  [|a: O, b: O| a + b]        [None::<fn(O) -> O>];
    [uncontiguous_reducel2_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x * x}]                  [|x: O| x * x]                  [|a: O, b: O| a + b]        [None::<fn(O) -> O>];

    [uncontiguous_reducel3_f32]   [f32]     [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))];
    [uncontiguous_reducel3_f16]   [f16]     [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))];
    [uncontiguous_reducel3_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x.abs().powi(3)]                                [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))];
    [uncontiguous_reducel3_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.abs().powi(3)}]        [|x: O| x.abs().powi(3)]        [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))];
    [uncontiguous_reducel3_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.abs().powi(3)}]        [|x: O| x.abs().powi(3)]        [|a: O, b: O| a + b]        [Some(|x: O| x.powf(O::ONE/O::THREE))];

    [uncontiguous_logsumexp_f32]   [f32]     [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())];
    [uncontiguous_logsumexp_f16]   [f16]     [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())];
    [uncontiguous_logsumexp_bf16]  [bf16]    [FloatOutUnary]   [|x: T| x.exp()]                                       [preop]                         [|a: O, b: O| a + b]        [Some(|x: O| x.ln())];
    [uncontiguous_logsumexp_i8]    [i8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.exp()}]               [|x: O| x.exp()]                [|a: O, b: O| a + b]        [Some(|x: O| x.ln())];
    [uncontiguous_logsumexp_u8]    [u8]      [FloatOutUnary]   [|x: T| {let x = f16::from(x); x.exp()}]               [|x: O| x.exp()]                [|a: O, b: O| a + b]        [Some(|x: O| x.ln())];
)]
pub(crate) fn func_name(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
) -> std::result::Result<Tensor, TensorError>
{
    type T = in_type;
    type O = <T as trait_name>::Output;
    assert_eq!(res_dtype, O::to_dtype());
    let preop = pre_op;
    let cumulate = cumulate_op;
    let postop = post_op;
    uncontiguos_reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        res_dtype,
        c,
        move |res| {
            let res = res as *mut O;
            let init_val: O = init_val.cast();
            let val = a
                .par_iter()
                .par_strided_fold(init_val, |acc, x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                unsafe { *res = postop(val); }
            } else {
                unsafe { *res = val; }
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            let iterators = UCReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs;
                let a_data_ptr = iterator.ptrs;
                let current_size = iterator.end - iterator.start;
                let res_shape = res_shape.clone();
                let res_strides = result.strides().clone();
                let shape_len = iterator.a_shape.len() as i64;

                uncontiguous_reduce_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
        move |num_threads, inner_loop_size, ap, result| {
            let intervals = mt_intervals(inner_loop_size, num_threads);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); ap.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[ap.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(ap.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, mut res)| {
                    res.iter_mut().zip(inp.iter()).for_each(|(x, y)| {
                        *x = cumulate(*x, preop(y));
                    });
                    if let Some(postop) = postop {
                        res.iter_mut().for_each(|x| {
                            *x = postop(*x);
                        });
                    }
                });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let outer_loop_size = result.size() / inner_loop_size;
            let iterators = UCReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let a_last_stride = transposed_tensor.strides()[a.ndim() - axes.len() - 1];
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let res_last_strides = *result.strides().inner().last().unwrap();
                let res_strides = result.strides().clone();
                let res_shape = res_shape.clone();
                let shape_len = iterator.shape.len() as i64;
                uncontiguous_reduce_dim_not_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.a_prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    res_last_strides as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
    )
}

impl Tensor {

    #[duplicate::duplicate_item(
        func_name     promote_func              description                                                         test_code;
        [sum]         [promote_normal_unary]    ["Computes sum along the specified axes."]                          ["let y = x.sum(&[1], true)?;"];
        [prod]        [promote_normal_unary]    ["Computes product along the specified axes."]                      ["let y = x.prod(&[1], true)?;"];
        [nansum]      [promote_normal_unary]    ["Computes sum along the specified axes, ignoring NaNs."]           ["let y = x.nansum(&[1], true)?;"];
        [nanprod]     [promote_normal_unary]    ["Computes product along the specified axes, ignoring NaNs."]       ["let y = x.nanprod(&[1], true)?;"];
        [min]         [promote_normal_unary]    ["Computes minimum along the specified axes."]                      ["let y = x.min(&[1], true)?;"];
        [max]         [promote_normal_unary]    ["Computes maximum along the specified axes."]                      ["let y = x.max(&[1], true)?;"];
        [sumsquare]   [promote_normal_unary]    ["Computes sum of squares along the specified axes."]               ["let y = x.sumsquare(&[1], true)?;"];
        [reducel1]    [promote_normal_unary]    ["Computes L1 norm along the specified axes."]                      ["let y = x.reducel1(&[1], true)?;"];

        [all]         [promote_eval]            ["Computes all along the specified axes."]                          ["let y = x.all(&[1], true)?;"];
        [any]         [promote_eval]            ["Computes any along the specified axes."]                          ["let y = x.any(&[1], true)?;"];

        [logsumexp]   [promote_float_unary]     ["Computes logsumexp along the specified axes."]                    ["let y = x.logsumexp(&[1], true)?;"];
        [reducel2]    [promote_float_unary]     ["Computes L2 norm along the specified axes."]                      ["let y = x.reducel2(&[1], true)?;"];
        [reducel3]    [promote_float_unary]     ["Computes L3 norm along the specified axes."]                      ["let y = x.reducel3(&[1], true)?;"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    /// 
    /// `dims`: Dimensions to reduce over
    /// 
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    /// 
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    /// 
    /// # Returns:
    /// A new tensor with the reduced dimensions.
    pub fn func_name(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        let res_dtype = promote_func(self.dtype);
        if self.is_contiguous() && self.parent.is_none() {
            match self.dtype {
                DType::F32 => {
                    paste::paste!(
                        [<contiguous_ func_name _f32>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::F16 => {
                    paste::paste!(
                        [<contiguous_ func_name _f16>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::BF16 => {
                    paste::paste!(
                        [<contiguous_ func_name _bf16>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::I8 => {
                    paste::paste!(
                        [<contiguous_ func_name _i8>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::U8 => {
                    paste::paste!(
                        [<contiguous_ func_name _u8>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                _ => {
                    unimplemented!()
                }
            }
        } else {
            match self.dtype {
                DType::F32 => {
                    paste::paste!(
                        [<uncontiguous_ func_name _f32>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::F16 => {
                    paste::paste!(
                        [<uncontiguous_ func_name _f16>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::BF16 => {
                    paste::paste!(
                        [<uncontiguous_ func_name _bf16>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::I8 => {
                    paste::paste!(
                        [<uncontiguous_ func_name _i8>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                DType::U8 => {
                    paste::paste!(
                        [<uncontiguous_ func_name _u8>](self, &axes, 0.0, keepdims, true, res_dtype, None)
                    )
                }
                _ => {
                    unimplemented!()
                }
            }
        }
    }

    #[duplicate::duplicate_item(
        func_name      method      promote_func                 description                                                         test_code;
        [sum_]         [sum]        [promote_normal_unary]      ["Computes sum along the specified axes."]                          ["let y = x.sum_(&[1], true, &mut out)?;"];
        [prod_]        [prod]       [promote_normal_unary]      ["Computes product along the specified axes."]                      ["let y = x.prod_(&[1], true, &mut out)?;"];
        [nansum_]      [nansum]     [promote_normal_unary]      ["Computes sum along the specified axes, ignoring NaNs."]           ["let y = x.nansum_(&[1], true, &mut out)?;"];
        [nanprod_]     [nanprod]    [promote_normal_unary]      ["Computes product along the specified axes, ignoring NaNs."]       ["let y = x.nanprod_(&[1], true, &mut out)?;"];
        [min_]         [min]        [promote_normal_unary]      ["Computes minimum along the specified axes."]                      ["let y = x.min_(&[1], true, &mut out)?;"];
        [max_]         [max]        [promote_normal_unary]      ["Computes maximum along the specified axes."]                      ["let y = x.max_(&[1], true, &mut out)?;"];
        [sumsquare_]   [sumsquare]  [promote_normal_unary]      ["Computes sum of squares along the specified axes."]               ["let y = x.sumsquare_(&[1], true, &mut out)?;"];
        [reducel1_]    [reducel1]   [promote_normal_unary]      ["Computes L1 norm along the specified axes."]                      ["let y = x.reducel1_(&[1], true, &mut out)?;"];

        [all_]         [all]        [promote_eval]              ["Computes all along the specified axes."]                          ["let y = x.all_(&[1], true, &mut out)?;"];
        [any_]         [any]        [promote_eval]              ["Computes any along the specified axes."]                          ["let y = x.any_(&[1], true, &mut out)?;"];

        [logsumexp_]   [logsumexp]  [promote_float_unary]       ["Computes logsumexp along the specified axes."]                    ["let y = x.logsumexp_(&[1], true, &mut out)?;"];
        [reducel2_]    [reducel2]   [promote_float_unary]       ["Computes L2 norm along the specified axes."]                      ["let y = x.reducel2_(&[1], true, &mut out)?;"];
        [reducel3_]    [reducel3]   [promote_float_unary]       ["Computes L3 norm along the specified axes."]                      ["let y = x.reducel3_(&[1], true, &mut out)?;"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    /// 
    /// `dims`: Dimensions to reduce over
    /// 
    /// `keepdim`: Whether to keep the reduced dimensions with length 1
    /// 
    /// `out`: tensor to store the result
    /// 
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    /// let mut out = Tensor::empty(/*shape*/&[2, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    /// 
    /// # Returns:
    /// A new tensor with the reduced dimensions.
    pub fn func_name(&self, axes: &[i64], keepdims: bool, out: &mut Tensor) -> Result<Tensor, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        let res_dtype = promote_func(self.dtype);
        if self.is_contiguous() && self.parent.is_none() {
            match self.dtype {
                DType::F32 => {
                    paste::paste!(
                        [<contiguous_ method _f32>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::F16 => {
                    paste::paste!(
                        [<contiguous_ method _f16>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::BF16 => {
                    paste::paste!(
                        [<contiguous_ method _bf16>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::I8 => {
                    paste::paste!(
                        [<contiguous_ method _i8>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::U8 => {
                    paste::paste!(
                        [<contiguous_ method _u8>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                _ => {
                    unimplemented!()
                }
            }
        } else {
            match self.dtype {
                DType::F32 => {
                    paste::paste!(
                        [<uncontiguous_ method _f32>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::F16 => {
                    paste::paste!(
                        [<uncontiguous_ method _f16>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::BF16 => {
                    paste::paste!(
                        [<uncontiguous_ method _bf16>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::I8 => {
                    paste::paste!(
                        [<uncontiguous_ method _i8>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                DType::U8 => {
                    paste::paste!(
                        [<uncontiguous_ method _u8>](self, &axes, 0.0, keepdims, true, res_dtype, Some(out.clone()))
                    )
                }
                _ => {
                    unimplemented!()
                }
            }
        }
    }
}