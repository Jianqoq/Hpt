use crate::{
    microkernel_trait::MatmulMicroKernel,
    template::{matmul_mp, matmul_mp_post},
    utils::kernel_params,
    vec_size,
    Pointer,
    PrePackedRhs,
    Zero,
    ALIGN,
};

pub(crate) fn matmul_template_no_block_info<T, TVec>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<&PrePackedRhs>
)
    where T: Zero + MatmulMicroKernel + Send + Sync + Copy
{
    let (nr, mr) = (T::get_max_nr() * vec_size::<T>(), T::get_max_mr());
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    let lhs_col_stride = lhs_strides[1];
    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>());
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul::<T, _, _>(
        Pointer::new(a.0 as *mut T, a.1),
        Pointer::new(b.0 as *mut T, b.1),
        Pointer::new(out.0 as *mut T, out.1),
        m,
        n,
        k,
        lhs_strides[0],
        rhs_strides[0],
        out_strides[0],
        lhs_strides[1],
        rhs_strides[1],
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        do_lhs_pack,
        num_threads,
        prepacked_rhs,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!()
    );
}

pub(crate) fn matmul_post_op_template_no_block_info<T, F1, F2>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<&PrePackedRhs>,
    post_op: F1,
    post_vec_op: F2
)
    where
        T: Zero + MatmulMicroKernel + Send + Sync + Copy,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec +
            Clone +
            Send +
            Sync +
            'static
{
    let (nr, mr) = (T::get_max_nr() * vec_size::<T>(), T::get_max_mr());
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    let lhs_col_stride = lhs_strides[1];
    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>());
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul_post::<T, _, _>(
        Pointer::new(a.0 as *mut T, a.1),
        Pointer::new(b.0 as *mut T, b.1),
        Pointer::new(out.0 as *mut T, out.1),
        m,
        n,
        k,
        lhs_strides[0],
        rhs_strides[0],
        out_strides[0],
        lhs_strides[1],
        rhs_strides[1],
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        do_lhs_pack,
        num_threads,
        prepacked_rhs,
        post_op,
        post_vec_op
    );
}

#[inline(always)]
pub fn matmul_mp_template_no_block_info<T>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    num_threads: usize,
    pack_vec: fn(
        *mut <T as MatmulMicroKernel>::MixedVec,
        *const <T as MatmulMicroKernel>::SelfVec,
        usize
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    pack_zero: fn(&mut <T as MatmulMicroKernel>::MixedType, &T),
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType)
)
    where
        T: MatmulMicroKernel + Send + Sync + Copy + Zero,
        <T as MatmulMicroKernel>::MixedType: Send + Sync + Copy + Zero
{
    let nr = T::get_max_mixed_precision_nr() * vec_size::<T>();
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>());
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.next_multiple_of(nr);
    }
    matmul_mp::<T, _, _>(
        Pointer::new(a.0 as *mut T, a.1),
        Pointer::new(b.0 as *mut T, b.1),
        Pointer::new(out.0 as *mut T, out.1),
        m,
        n,
        k,
        lhs_strides[0],
        rhs_strides[0],
        out_strides[0],
        lhs_strides[1],
        rhs_strides[1],
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
        None,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!(),
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back
    );
}

#[inline(always)]
pub fn matmul_post_op_mp_template_no_block_info<T, F1, F2>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    num_threads: usize,
    pack_vec: fn(
        *mut <T as MatmulMicroKernel>::MixedVec,
        *const <T as MatmulMicroKernel>::SelfVec,
        usize
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    pack_zero: fn(&mut <T as MatmulMicroKernel>::MixedType, &T),
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType),
    post_op: F1,
    post_vec_op: F2
)
    where
        T: MatmulMicroKernel + Send + Sync + Copy + Zero,
        <T as MatmulMicroKernel>::MixedType: Send + Sync + Copy + Zero,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec +
            Clone +
            Send +
            Sync +
            'static
{
    let nr = T::get_max_mixed_precision_nr() * vec_size::<T>();
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>());
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.next_multiple_of(nr);
    }
    matmul_mp_post::<T, _, _>(
        Pointer::new(a.0 as *mut T, a.1),
        Pointer::new(b.0 as *mut T, b.1),
        Pointer::new(out.0 as *mut T, out.1),
        m,
        n,
        k,
        lhs_strides[0],
        rhs_strides[0],
        out_strides[0],
        lhs_strides[1],
        rhs_strides[1],
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
        None,
        post_op,
        post_vec_op,
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back
    );
}

pub fn matmul<T: 'static>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<&PrePackedRhs>,
    post_op: Option<&(dyn (Fn(T, usize, usize) -> T) + Send + Sync)>,
    post_vec_op: Option<
        &(dyn (Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec) +
            Send +
            Sync)
    >
)
    where T: MatmulMicroKernel
{
    macro_rules! case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                match (post_op, post_vec_op) {
                    (Some(post_op), Some(post_vec_op)) => {
                        let post_op: &(dyn Fn($dtype, usize, usize) -> $dtype + Send + Sync) = unsafe { std::mem::transmute(post_op) };
                        let post_vec_op: &(dyn Fn($vec, usize, usize) -> $vec + Send + Sync) = unsafe { std::mem::transmute(post_vec_op) };
                        matmul_post_op_template_no_block_info::<$dtype, _, _>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            prepacked_rhs,
                            post_op,
                            post_vec_op
                        );
                    }
                    _ => {
                        matmul_template_no_block_info::<$dtype, $vec>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            prepacked_rhs,
                        );
                    }
                }
                return;
            }
        };
    }
    #[cfg(feature = "bool")]
    case!(bool, crate::BoolVec);
    #[cfg(feature = "f32")]
    case!(f32, crate::F32Vec);
    #[cfg(feature = "f64")]
    case!(f64, crate::F64Vec);
    #[cfg(feature = "i8")]
    case!(i8, crate::I8Vec);
    #[cfg(feature = "u8")]
    case!(u8, crate::U8Vec);
    #[cfg(feature = "i16")]
    case!(i16, crate::I16Vec);
    #[cfg(feature = "u16")]
    case!(u16, crate::U16Vec);
    #[cfg(feature = "i32")]
    case!(i32, crate::I32Vec);
    #[cfg(feature = "u32")]
    case!(u32, crate::U32Vec);
    #[cfg(feature = "i64")]
    case!(i64, crate::I64Vec);
    #[cfg(feature = "u64")]
    case!(u64, crate::U64Vec);
    #[cfg(feature = "cplx32")]
    case!(num_complex::Complex32, crate::Cplx32Vec);
    #[cfg(feature = "cplx64")]
    case!(num_complex::Complex64, crate::Cplx64Vec);

    #[cfg(any(feature = "f16", feature = "bf16"))]
    macro_rules! f16_case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                use crate::VecTrait;
                matmul_mp_template_no_block_info::<$dtype>(
                    (a.0 as *const $dtype, a.1),
                    (b.0 as *const $dtype, b.1),
                    (out.0 as *mut $dtype, out.1),
                    m,
                    n,
                    k,
                    lhs_strides,
                    rhs_strides,
                    out_strides,
                    num_threads,
                    |packed_b, b, i| unsafe {
                        let packed_b = packed_b as *mut crate::F32Vec;
                        let b = b as *const crate::F16Vec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        let b_vec = b.add(i).read_unaligned();
                        let val_f32 = b_vec.to_2_f32vec();
                        packed_b_vec0.write(val_f32[0]);
                        packed_b_vec1.write(val_f32[1]);
                    },
                    |packed_b, i| unsafe {
                        let packed_b = packed_b as *mut crate::F32Vec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        packed_b_vec0.write(crate::F32Vec::splat(0.0));
                        packed_b_vec1.write(crate::F32Vec::splat(0.0));
                    },
                    |im, val| {
                        let val = val as *const $dtype;
                        *im = unsafe { val.read().to_f32() };
                    },
                    |im, val| {
                        let im = im as *mut crate::F16Vec;
                        let val = val as *const crate::F32Vec;
                        let vec0 = unsafe { val.read() };
                        let vec1 = unsafe { val.add(1).read() };
                        unsafe { im.write_unaligned(crate::F16Vec::from_2_f32vec([vec0, vec1])) };
                    },
                    |im, val| {
                        let im = im as *mut $dtype;
                        unsafe { *im = <$dtype>::from_f32(*val) };
                    },
                );
            }
        };
    }
    #[cfg(feature = "f16")]
    f16_case!(half::f16, crate::F16Vec);
    #[cfg(feature = "bf16")]
    f16_case!(half::bf16, crate::Bf16Vec);
}

pub fn addmm<T: 'static>(
    a: (*const T, i64),
    b: (*const T, i64),
    out: (*mut T, i64),
    bias: (*const T, i64),
    m: usize,
    n: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    out_strides: [i64; 2],
    bias_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<&PrePackedRhs>,
    post_op: Option<&(dyn (Fn(T, usize, usize) -> T) + Send + Sync)>,
    post_vec_op: Option<
        &(dyn (Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec) +
            Send +
            Sync)
    >
)
    where T: MatmulMicroKernel
{
    macro_rules! case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                let bias_ptr = Pointer::new(bias.0 as *mut $dtype, bias.1);
                match (post_op, post_vec_op) {
                    (Some(post_op), Some(post_vec_op)) => {
                        let post_op: &(dyn Fn($dtype, usize, usize) -> $dtype + Send + Sync) = unsafe { std::mem::transmute(post_op) };
                        let post_vec_op: &(dyn Fn($vec, usize, usize) -> $vec + Send + Sync) = unsafe { std::mem::transmute(post_vec_op) };
                        matmul_post_op_template_no_block_info::<$dtype, _, _>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            prepacked_rhs,
                            move |res, m, n| {
                                let bias = bias_ptr[m as i64 * bias_strides[0] + n as i64 * bias_strides[1]];
                                post_op(res + bias, m, n)
                            },
                            move |res, m, n| {
                                let bias = bias_ptr.offset(m as i64 * bias_strides[0] + n as i64 * bias_strides[1]);
                                post_vec_op(res + bias.cast::<$vec>().read_unaligned(), m, n)
                            }
                        );
                    }
                    _ => {
                        matmul_post_op_template_no_block_info::<$dtype, _, _>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            prepacked_rhs,
                            move |res, m, n| {
                                let bias = bias_ptr[m as i64 * bias_strides[0] + n as i64 * bias_strides[1]];
                                res + bias
                            },
                            move |res, m, n| {
                                let bias = bias_ptr.offset(m as i64 * bias_strides[0] + n as i64 * bias_strides[1]);
                                res + bias.cast::<$vec>().read_unaligned()
                            }
                        );
                    }
                }
                return;
            }
        };
    }
    #[cfg(feature = "bool")]
    case!(bool, crate::BoolVec);
    #[cfg(feature = "f32")]
    case!(f32, crate::F32Vec);
    #[cfg(feature = "f64")]
    case!(f64, crate::F64Vec);
    #[cfg(feature = "i8")]
    case!(i8, crate::I8Vec);
    #[cfg(feature = "u8")]
    case!(u8, crate::U8Vec);
    #[cfg(feature = "i16")]
    case!(i16, crate::I16Vec);
    #[cfg(feature = "u16")]
    case!(u16, crate::U16Vec);
    #[cfg(feature = "i32")]
    case!(i32, crate::I32Vec);
    #[cfg(feature = "u32")]
    case!(u32, crate::U32Vec);
    #[cfg(feature = "i64")]
    case!(i64, crate::I64Vec);
    #[cfg(feature = "u64")]
    case!(u64, crate::U64Vec);
    #[cfg(feature = "cplx32")]
    case!(num_complex::Complex32, crate::Cplx32Vec);
    #[cfg(feature = "cplx64")]
    case!(num_complex::Complex64, crate::Cplx64Vec);

    #[cfg(any(feature = "f16", feature = "bf16"))]
    macro_rules! f16_case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                use crate::VecTrait;
                let bias_ptr = Pointer::new(bias.0 as *mut $dtype, bias.1);
                match (post_op, post_vec_op) {
                    (Some(post_op), Some(post_vec_op)) => {
                        let post_op: &(dyn Fn($dtype, usize, usize) -> $dtype + Send + Sync) = unsafe { std::mem::transmute(post_op) };
                        let post_vec_op: &(dyn Fn($vec, usize, usize) -> $vec + Send + Sync) = unsafe { std::mem::transmute(post_vec_op) };
                        matmul_post_op_mp_template_no_block_info::<$dtype, _, _>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            |packed_b, b, i| unsafe {
                                let packed_b = packed_b as *mut crate::F32Vec;
                                let b = b as *const crate::F16Vec;
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                let b_vec = b.add(i).read_unaligned();
                                let val_f32 = b_vec.to_2_f32vec();
                                packed_b_vec0.write(val_f32[0]);
                                packed_b_vec1.write(val_f32[1]);
                            },
                            |packed_b, i| unsafe {
                                let packed_b = packed_b as *mut crate::F32Vec;
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                packed_b_vec0.write(crate::F32Vec::splat(0.0));
                                packed_b_vec1.write(crate::F32Vec::splat(0.0));
                            },
                            |im, val| {
                                let val = val as *const $dtype;
                                *im = unsafe { val.read().to_f32() };
                            },
                            |im, val| {
                                let im = im as *mut crate::F16Vec;
                                let val = val as *const crate::F32Vec;
                                let vec0 = unsafe { val.read() };
                                let vec1 = unsafe { val.add(1).read() };
                                unsafe { im.write_unaligned(crate::F16Vec::from_2_f32vec([vec0, vec1])) };
                            },
                            |im, val| {
                                let im = im as *mut $dtype;
                                unsafe { *im = <$dtype>::from_f32(*val) };
                            },
                            move |res, m, n| {
                                let bias = bias_ptr[m as i64 * bias_strides[0] + n as i64 * bias_strides[1]];
                                post_op(res + bias, m, n)
                            },
                            move |res, m, n| {
                                let bias = bias_ptr.offset(m as i64 * bias_strides[0] + n as i64 * bias_strides[1]);
                                post_vec_op(res + bias.cast::<$vec>().read_unaligned(), m, n)
                            }
                        );
                    },
                    _ => {
                        matmul_post_op_mp_template_no_block_info::<$dtype, _, _>(
                            (a.0 as *const $dtype, a.1),
                            (b.0 as *const $dtype, b.1),
                            (out.0 as *mut $dtype, out.1),
                            m,
                            n,
                            k,
                            lhs_strides,
                            rhs_strides,
                            out_strides,
                            num_threads,
                            |packed_b, b, i| unsafe {
                                let packed_b = packed_b as *mut crate::F32Vec;
                                let b = b as *const crate::F16Vec;
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                let b_vec = b.add(i).read_unaligned();
                                let val_f32 = b_vec.to_2_f32vec();
                                packed_b_vec0.write(val_f32[0]);
                                packed_b_vec1.write(val_f32[1]);
                            },
                            |packed_b, i| unsafe {
                                let packed_b = packed_b as *mut crate::F32Vec;
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                packed_b_vec0.write(crate::F32Vec::splat(0.0));
                                packed_b_vec1.write(crate::F32Vec::splat(0.0));
                            },
                            |im, val| {
                                let val = val as *const $dtype;
                                *im = unsafe { val.read().to_f32() };
                            },
                            |im, val| {
                                let im = im as *mut crate::F16Vec;
                                let val = val as *const crate::F32Vec;
                                let vec0 = unsafe { val.read() };
                                let vec1 = unsafe { val.add(1).read() };
                                unsafe { im.write_unaligned(crate::F16Vec::from_2_f32vec([vec0, vec1])) };
                            },
                            |im, val| {
                                let im = im as *mut $dtype;
                                unsafe { *im = <$dtype>::from_f32(*val) };
                            },
                            move |res, m, n| {
                                let bias = bias_ptr[m as i64 * bias_strides[0] + n as i64 * bias_strides[1]];
                                res + bias
                            },
                            move |res, m, n| {
                                let bias = bias_ptr.offset(m as i64 * bias_strides[0] + n as i64 * bias_strides[1]);
                                res + bias.cast::<$vec>().read_unaligned()
                            }
                        );
                    }
                }
            }
        };
    }
    #[cfg(feature = "f16")]
    f16_case!(half::f16, crate::F16Vec);
    #[cfg(feature = "bf16")]
    f16_case!(half::bf16, crate::Bf16Vec);
}

pub fn prepack_rhs<T>(
    b: (*const T, i64),
    n: usize,
    k: usize,
    rhs_strides: [i64; 2],
    parallel: bool
) -> PrePackedRhs
    where T: MatmulMicroKernel + 'static + Send + Sync
{
    let (nr, mr) = (T::get_max_nr() * vec_size::<T>(), T::get_max_mr());

    let mut param = kernel_params(n, 0, k, nr, mr, std::mem::size_of::<T>());
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }

    let mem_size = crate::template::prepack_b_size::<T>(n, k, param.kc, nr);

    let layout = std::alloc::Layout::from_size_align(mem_size, ALIGN).unwrap();

    let raw_buffer = unsafe { std::alloc::alloc(layout) as *mut T };

    let buffer = Pointer::new(raw_buffer, (mem_size / std::mem::size_of::<T>()) as i64);

    crate::template::prepack_b(
        Pointer::new(b.0 as *mut T, b.1),
        buffer,
        n,
        k,
        rhs_strides,
        param.kc,
        param.nc,
        nr,
        parallel
    );

    PrePackedRhs {
        buffer: (buffer.cast::<u8>(), layout),
        kc: param.kc,
        nr,
        nc: param.nc,
    }
}
