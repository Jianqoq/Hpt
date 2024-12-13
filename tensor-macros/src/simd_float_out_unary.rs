use crate::type_utils::type_simd_is_arr;
use crate::type_utils::Type;
use crate::type_utils::{type_simd_lanes, SimdType, TypeInfo};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_float_out_unary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        ("bool", type_simd_lanes("bool"), "bool"),
        ("bf16", type_simd_lanes("bf16"), "bf16"),
        ("f16", type_simd_lanes("f16"), "f16"),
        ("f32", type_simd_lanes("f32"), "f32"),
        ("f64", type_simd_lanes("f64"), "f64"),
        ("i8", type_simd_lanes("i8"), "i8"),
        ("i16", type_simd_lanes("i16"), "i16"),
        ("i32", type_simd_lanes("i32"), "i32"),
        ("i64", type_simd_lanes("i64"), "i64"),
        ("u8", type_simd_lanes("u8"), "u8"),
        ("u16", type_simd_lanes("u16"), "u16"),
        ("u32", type_simd_lanes("u32"), "u32"),
        ("u64", type_simd_lanes("u64"), "u64"),
        ("isize", type_simd_lanes("isize"), "isize"),
        ("usize", type_simd_lanes("usize"), "usize"),
        ("Complex32", type_simd_lanes("complex32"), "complex32"),
        ("Complex64", type_simd_lanes("complex64"), "complex64"),
    ];

    for (lhs_ty, _, lhs) in types.iter() {
        let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
        let res_type = lhs_type.infer_float_res_type_uary();
        let lhs_lanes = type_simd_lanes(lhs);
        let lhs_simd: SimdType = (*lhs).into();
        let res_simd_ty = Ident::new(
            &format!(
                "{}x{}",
                if res_type.is_cplx() {
                    match res_type.to_string().as_str() {
                        "complex32" => "cplx32".to_string(),
                        "complex64" => "cplx64".to_string(),
                        _ => unreachable!(),
                    }
                } else {
                    res_type.to_string()
                },
                type_simd_lanes(&res_type.to_string())
            ),
            proc_macro2::Span::call_site(),
        );
        let to_res_type = Ident::new(
            &format!("to_{}", res_type.to_string().to_lowercase()),
            proc_macro2::Span::call_site(),
        );
        let res_lanes = type_simd_lanes(&res_type.to_string());
        if res_lanes != lhs_lanes {
            let res = unreachable_impl(lhs_simd, res_type, res_simd_ty.clone());
            ret.extend(res);
            continue;
        }
        if type_simd_is_arr(lhs) || type_simd_is_arr(&res_type.to_string()) {
            let sin = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_sin", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let cos = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_cos", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let tan = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_tan", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let asin = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_asin", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let acos = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_acos", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let atan = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_atan", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let sinh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_sinh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let cosh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_cosh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let tanh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_tanh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let asinh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_asinh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let acosh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_acosh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let atanh = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_atanh", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let recip = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_recip", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let erf = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_erf", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let sigmoid = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_sigmoid", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let ln = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_ln", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let exp = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_exp", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let exp2 = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_exp2", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let log2 = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_log2", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let log10 = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_log10", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let sqrt = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_sqrt", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let gelu = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_gelu", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let _hard_swish = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_hard_swish", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let soft_plus = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_softplus", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let softsign = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_softsign", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let mish = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_mish", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let cbrt = gen_func_arr(
                lhs_type,
                lhs_lanes,
                res_type,
                res_simd_ty.clone(),
                Ident::new("_cbrt", proc_macro2::Span::call_site()),
                to_res_type.clone(),
            );
            let celu = {
                let unroll = (0..lhs_lanes as usize).map(|i| {
                    quote! {
                        arr[#i] = self_arr[#i]._celu(alpha);
                    }
                });
                let lhs_lanes = lhs_lanes as usize;
                quote! {
                    fn _celu(self, alpha: Self::Base) -> Self::Output {
                        let mut arr = [#res_type::ZERO; #lhs_lanes];
                        let self_arr: [#res_type; #lhs_lanes] = unsafe { std::mem::transmute(self) };
                        #(#unroll)*
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let selu = {
                let unroll = (0..lhs_lanes as usize).map(|i| {
                    quote! {
                        arr[#i] = self_arr[#i]._selu(alpha, scale);
                    }
                });
                let lhs_lanes = lhs_lanes as usize;
                quote! {
                    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
                        let mut arr = [#res_type::ZERO; #lhs_lanes];
                        let self_arr: [#res_type; #lhs_lanes] = unsafe { std::mem::transmute(self) };
                        #(#unroll)*
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let elu = {
                let unroll = (0..lhs_lanes as usize).map(|i| {
                    quote! {
                        arr[#i] = self_arr[#i]._elu(alpha);
                    }
                });
                let lhs_lanes = lhs_lanes as usize;
                quote! {
                    fn _elu(self, alpha: Self::Base) -> Self::Output {
                        let mut arr = [#res_type::ZERO; #lhs_lanes];
                        let self_arr: [#res_type; #lhs_lanes] = unsafe { std::mem::transmute(self) };
                        #(#unroll)*
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let hard_sigmoid = {
                let unroll = (0..lhs_lanes as usize).map(|i| {
                    quote! {
                        arr[#i] = self_arr[#i]._hard_sigmoid();
                    }
                });
                let lhs_lanes = lhs_lanes as usize;
                quote! {
                    fn _hard_sigmoid(self) -> Self::Output {
                        let mut arr = [#res_type::ZERO; #lhs_lanes];
                        let self_arr: [#res_type; #lhs_lanes] = unsafe { std::mem::transmute(self) };
                        #(#unroll)*
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let fast_hard_sigmoid = {
                let unroll = (0..lhs_lanes as usize).map(|i| {
                    quote! {
                        arr[#i] = self_arr[#i]._fast_hard_sigmoid();
                    }
                });
                let lhs_lanes = lhs_lanes as usize;
                quote! {
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        let mut arr = [#res_type::ZERO; #lhs_lanes];
                        let self_arr: [#res_type; #lhs_lanes] = unsafe { std::mem::transmute(self) };
                        #(#unroll)*
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let res = quote! {
                impl FloatOutUnary for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    type Base = #res_type;
                    #exp #exp2 #ln #log2 #log10 #sqrt #sin #cos #tan #asin #acos #atan #sinh #cosh #tanh #asinh #acosh
                    #atanh #recip #erf #sigmoid #gelu #_hard_swish #soft_plus #softsign #mish #cbrt
                    #celu #selu #elu #hard_sigmoid #fast_hard_sigmoid
                }
            };
            ret.extend(res);
        } else {
            let non_sleef = |func_name: &str, non_sleef_func: &str| {
                let func_name = Ident::new(func_name, proc_macro2::Span::call_site());
                let non_sleef_func = Ident::new(non_sleef_func, proc_macro2::Span::call_site());
                quote! {
                    #[inline(always)]
                    fn #func_name(self) -> Self::Output {
                        SimdMath::#non_sleef_func(self.#to_res_type())
                    }
                }
            };
            let sin = non_sleef("_sin", "sin");
            let cos = non_sleef("_cos", "cos");
            let tan = non_sleef("_tan", "tan");
            let asin = non_sleef("_asin", "asin");
            let acos = non_sleef("_acos", "acos");
            let atan = non_sleef("_atan", "atan");
            let sinh = non_sleef("_sinh", "sinh");
            let cosh = non_sleef("_cosh", "cosh");
            let tanh = non_sleef("_tanh", "tanh");
            let asinh = non_sleef("_asinh", "asinh");
            let atanh = non_sleef("_atanh", "atanh");
            let acosh = non_sleef("_acosh", "acosh");
            let ln = non_sleef("_ln", "log");
            let exp = non_sleef("_exp", "exp");
            let exp2 = non_sleef("_exp2", "exp2");
            let log2 = non_sleef("_log2", "log2");
            let log10 = non_sleef("_log10", "log10");
            let sqrt = non_sleef("_sqrt", "sqrt");
            let cbrt = non_sleef("_cbrt", "cbrt");
            let erf = non_sleef("_erf", "erf");
            let res = quote! {
                impl FloatOutUnary for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    type Base = #res_type;
                    #sin #cos #tan #asin #acos #atan #sinh #cosh #tanh #asinh #erf #ln #exp #log2 #log10 #sqrt
                    #cbrt #atanh #acosh #exp2
                    #[inline(always)]
                    fn _recip(self) -> Self::Output {
                        self.#to_res_type().recip()
                    }
                    #[inline(always)]
                    fn _sigmoid(self) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(#res_type::ONE) / (
                            #res_simd_ty::#res_simd_ty::splat(#res_type::ONE) + (-self.#to_res_type())._exp())
                    }
                    #[inline(always)]
                    fn _gelu(self) -> Self::Output {
                        let x = self.#to_res_type();
                        let erf = (x * #res_simd_ty::#res_simd_ty::splat(#res_type::FRAC_1_SQRT_2))._erf() + #res_simd_ty::#res_simd_ty::splat(#res_type::ONE);
                        let half = #res_simd_ty::#res_simd_ty::splat(#res_type::HALF);
                        half * x * erf
                    }
                    #[inline(always)]
                    fn _softplus(self) -> Self::Output {
                        let one = #res_simd_ty::#res_simd_ty::splat(#res_type::ONE);
                        (one + self._exp())._ln()
                    }
                    #[inline(always)]
                    fn _softsign(self) -> Self::Output {
                        let casted = self.#to_res_type();
                        casted / (#res_simd_ty::#res_simd_ty::splat(#res_type::ONE) + casted._abs())
                    }
                    #[inline(always)]
                    fn _mish(self) -> Self::Output {
                        let x = self.#to_res_type();
                        x * x._softplus()._tanh()
                    }
                    #[inline(always)]
                    fn _celu(self, alpha: Self::Base) -> Self::Output {
                        let x = self.#to_res_type();
                        let scale = #res_simd_ty::#res_simd_ty::splat(alpha);
                        let gt_mask = x._gt(#res_simd_ty::#res_simd_ty::splat(#res_type::ZERO));
                        gt_mask.select(
                            x,
                            scale * (x._exp() - #res_simd_ty::#res_simd_ty::splat(#res_type::ONE))
                        )
                    }
                    #[inline(always)]
                    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
                        let scale = #res_simd_ty::#res_simd_ty::splat(scale);
                        scale * self._elu(alpha)
                    }
                    #[inline(always)]
                    fn _elu(self, alpha: Self::Base) -> Self::Output {
                        let x = self.#to_res_type();
                        let alpha = #res_simd_ty::#res_simd_ty::splat(alpha);
                        let mask = x._gt(#res_simd_ty::#res_simd_ty::splat(#res_type::ZERO));
                        mask.select(x, alpha * (SimdMath::expm1(x)))
                    }
                    #[inline(always)]
                    fn _hard_swish(self) -> Self::Output {
                        let x = self.#to_res_type();
                        let six = #res_simd_ty::#res_simd_ty::splat(#res_type::SIX);
                        let three = #res_simd_ty::#res_simd_ty::splat(#res_type::THREE);
                        x * (x + three)._relu6() / six
                    }
                    #[inline(always)]
                    fn _hard_sigmoid(self) -> Self::Output {
                        let x = self.#to_res_type();
                        let point_two = #res_simd_ty::#res_simd_ty::splat(#res_type::POINT_TWO);
                        let half = #res_simd_ty::#res_simd_ty::splat(#res_type::HALF);
                        let one = #res_simd_ty::#res_simd_ty::splat(#res_type::ONE);
                        let zero = #res_simd_ty::#res_simd_ty::splat(#res_type::ZERO);
                        let add = point_two * x + half;
                        add._min(one)._max(zero)
                    }
                    #[inline(always)]
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        let x = self.#to_res_type();
                        let sixth = #res_simd_ty::#res_simd_ty::splat(#res_type::ONE / #res_type::SIX);
                        let half = #res_simd_ty::#res_simd_ty::splat(#res_type::HALF);
                        let one = #res_simd_ty::#res_simd_ty::splat(#res_type::ONE);
                        let zero = #res_simd_ty::#res_simd_ty::splat(#res_type::ZERO);
                        let result = x * sixth + half;
                        result._min(one)._max(zero)
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn gen_func_arr(
    lhs_dtype: TypeInfo,
    lhs_lanes: u8,
    res_type: Type,
    res_simd_ty: Ident,
    method: Ident,
    to_res_type: Ident,
) -> proc_macro2::TokenStream {
    let unroll = (0..lhs_lanes as usize).map(|i| {
        quote! {
            arr[#i] = self_arr[#i].#to_res_type().#method();
        }
    });
    let func = if lhs_dtype.dtype.is_f16() || lhs_dtype.dtype.is_bf16() {
        let f32_lanes = type_simd_lanes("f32");
        let ident = Ident::new(
            &format!("f32x{}", f32_lanes),
            proc_macro2::Span::call_site(),
        );
        if lhs_dtype.dtype.is_f16() {
            let f32_lanes = f32_lanes as usize;
            quote! {
                fn #method(self) -> Self::Output {
                    let mut arr = [f32::ZERO; #f32_lanes];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = self[i].to_f32();
                    }
                    let arr: #ident::#ident = unsafe { std::mem::transmute(arr) };
                    let res_f32 = arr.#method();
                    let mut half_arr = [half::f16::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        half_arr[i] = half::f16::from_f32(res_f32[i]);
                    }
                    #res_simd_ty::#res_simd_ty(half_arr)
                }
            }
        } else {
            quote! {
                fn #method(self) -> Self::Output {
                    #[cfg(target_feature = "avx2")]
                    let mut casted = self.to_2_f32x8();
                    #[cfg(all(any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"), not(target_feature = "avx2")))]
                    let mut casted = self.to_2_f32x4();
                    #[cfg(target_feature = "avx512f")]
                    let mut casted = self.to_2_f32x16();
                    for i in 0..2 {
                        casted[i] = casted[i].#method();
                    }
                    #[cfg(target_feature = "avx2")]
                    let ret = bf16x16::bf16x16::from_2_f32x8(casted);
                    #[cfg(all(any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"), not(target_feature = "avx2")))]
                    let ret = bf16x8::bf16x8::from_2_f32x4(casted);
                    #[cfg(target_feature = "avx512f")]
                    let ret = bf16x32::bf16x32::from_2_f32x16(casted);
                    ret
                }
            }
        }
    } else {
        quote! {
            fn #method(self) -> Self::Output {
                let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                let self_arr = self;
                #(#unroll)*
                #res_simd_ty::#res_simd_ty(arr.into())
            }
        }
    };
    func
}

fn unreachable_impl(
    lhs_simd: SimdType,
    res_ty: Type,
    res_simd_ty: Ident,
) -> proc_macro2::TokenStream {
    quote! {
        impl FloatOutUnary for #lhs_simd {
            type Output = #res_simd_ty::#res_simd_ty;
            type Base = #res_ty;
            fn _exp(self) -> Self::Output {unreachable!()}
            fn _exp2(self) -> Self::Output {unreachable!()}
            fn _ln(self) -> Self::Output {unreachable!()}
            fn _celu(self, alpha: Self::Base) -> Self::Output {unreachable!()}
            fn _log2(self) -> Self::Output {unreachable!()}
            fn _log10(self) -> Self::Output {unreachable!()}
            fn _sqrt(self) -> Self::Output {unreachable!()}
            fn _sin(self) -> Self::Output {unreachable!()}
            fn _cos(self) -> Self::Output {unreachable!()}
            fn _tan(self) -> Self::Output {unreachable!()}
            fn _asin(self) -> Self::Output {unreachable!()}
            fn _acos(self) -> Self::Output {unreachable!()}
            fn _atan(self) -> Self::Output {unreachable!()}
            fn _sinh(self) -> Self::Output {unreachable!()}
            fn _cosh(self) -> Self::Output {unreachable!()}
            fn _tanh(self) -> Self::Output {unreachable!()}
            fn _asinh(self) -> Self::Output {unreachable!()}
            fn _acosh(self) -> Self::Output {unreachable!()}
            fn _atanh(self) -> Self::Output {unreachable!()}
            fn _recip(self) -> Self::Output {unreachable!()}
            fn _erf(self) -> Self::Output {unreachable!()}
            fn _sigmoid(self) -> Self::Output {unreachable!()}
            fn _elu(self, alpha: Self::Base) -> Self::Output {unreachable!()}
            fn _gelu(self) -> Self::Output {unreachable!()}
            fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {unreachable!()}
            fn _hard_sigmoid(self) -> Self::Output {unreachable!()}
            fn _fast_hard_sigmoid(self) -> Self::Output {unreachable!()}
            fn _hard_swish(self) -> Self::Output {unreachable!()}
            fn _softplus(self) -> Self::Output {unreachable!()}
            fn _softsign(self) -> Self::Output {unreachable!()}
            fn _mish(self) -> Self::Output {unreachable!()}
            fn _cbrt(self) -> Self::Output {unreachable!()}
        }
    }
}
