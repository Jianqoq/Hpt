use hpt_traits::{
    ops::conv::{Conv, ConvBatchNorm},
    tensor::CommonBounds,
};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOutPromote},
};

use crate::{
    backends::cpu::kernels::{
        conv2d::{
            self, batchnorm_conv2d::batchnorm_conv2d, conv2d_group::conv2d_group, conv2d_new_mp,
            dwconv2d::dwconv2d, microkernel_trait::Conv2dMicroKernel,
        },
        matmul::microkernel_trait::MatmulMicroKernel,
    },
    tensor_base::_Tensor,
};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_types::traits::VecTrait;

impl<T, const DEVICE: usize, Al> Conv<T> for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds
        + Conv2dMicroKernel
        + Cast<<T as NormalOutPromote>::Intermediate>
        + MatmulMicroKernel,
    <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        post_scalar: Option<fn(T) -> T>,
        post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        if T::STR == "bf16" {
            type F32Vec = <<half::bf16 as NormalOutPromote>::Intermediate as TypeCommon>::Vec;
            type BF16Vec = <half::bf16 as TypeCommon>::Vec;
            let inp = self
                .static_cast::<half::bf16>()
                .expect("static_cast bf16 failed");
            let ker = kernels
                .static_cast::<half::bf16>()
                .expect("static_cast bf16 failed");
            let bias = bias.map(|x| {
                x.static_cast::<half::bf16>()
                    .expect("static_cast bf16 failed")
            });
            let res = conv2d_new_mp::conv2d::<half::bf16, DEVICE, Al>(
                &inp,
                &ker,
                bias.as_ref(),
                steps,
                padding,
                dilation,
                |x| {
                    let vec0 = unsafe { x.read_unaligned() };
                    let vec1 = unsafe { x.add(1).read_unaligned() };
                    BF16Vec::from_2_f32vec([vec0, vec1])
                },
                |x| unsafe {
                    let mut bf16_vec = BF16Vec::splat(half::bf16::from_f32_const(0.0));
                    for j in 0..F32Vec::SIZE {
                        bf16_vec[j] = *x.add(j);
                    }
                    let val_f32 = bf16_vec.high_to_f32vec();
                    val_f32
                },
                |x| x.cast(),
                |x| x.cast(),
                unsafe { std::mem::transmute(post_scalar) },
                unsafe { std::mem::transmute(post_vec) },
            )?;
            Ok(res.static_cast::<T>()?)
        } else if T::STR == "f16" && !cfg!(target_feature = "neon") {
            type F32Vec = <<half::f16 as NormalOutPromote>::Intermediate as TypeCommon>::Vec;
            type F16Vec = <half::f16 as TypeCommon>::Vec;
            let inp = self
                .static_cast::<half::f16>()
                .expect("static_cast f16 failed");
            let ker = kernels
                .static_cast::<half::f16>()
                .expect("static_cast f16 failed");
            let bias = bias.map(|x| {
                x.static_cast::<half::f16>()
                    .expect("static_cast f16 failed")
            });
            let res = conv2d_new_mp::conv2d::<half::f16, DEVICE, Al>(
                &inp,
                &ker,
                bias.as_ref(),
                steps,
                padding,
                dilation,
                |x| {
                    let vec0 = unsafe { x.read_unaligned() };
                    let vec1 = unsafe { x.add(1).read_unaligned() };
                    F16Vec::from_2_f32vec([vec0, vec1])
                },
                |x| unsafe {
                    let mut f16_vec = F16Vec::splat(half::f16::from_f32_const(0.0));
                    for j in 0..F32Vec::SIZE {
                        f16_vec[j] = *x.add(j);
                    }
                    let val_f32 = f16_vec.high_to_f32vec();
                    val_f32
                },
                |x| x.cast(),
                |x| x.cast(),
                unsafe { std::mem::transmute(post_scalar) },
                unsafe { std::mem::transmute(post_vec) },
            )?;
            Ok(res.static_cast::<T>()?)
        } else {
            conv2d::conv2d::conv2d(
                self,
                kernels,
                bias,
                steps,
                padding,
                dilation,
                post_scalar,
                post_vec,
            )
        }
    }

    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        post_scalar: Option<fn(T) -> T>,
        post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d_group(
            self,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            groups,
            post_scalar,
            post_vec,
        )
    }

    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        post_scalar: Option<fn(T) -> T>,
        post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        dwconv2d(
            self,
            bias,
            kernels,
            steps,
            padding,
            dilation,
            post_scalar,
            post_vec,
        )
    }

    fn conv2d_transpose(
        &self,
        _: &Self::Output,
        _: [i64; 2],
        _: [(i64, i64); 2],
        _: [i64; 2],
        _: [i64; 2],
        _: Option<fn(T) -> T>,
        _: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        unimplemented!()
    }
}

impl<T, const DEVICE: usize, A> ConvBatchNorm<T> for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds
        + Conv2dMicroKernel
        + MatmulMicroKernel
        + Cast<<T as NormalOutPromote>::Intermediate>,
    <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, A>;
    fn batchnorm_conv2d(
        &self,
        kernels: &Self::Output,
        mean: &Self::Output,
        var: &Self::Output,
        gamma: &Self::Output,
        beta: &Self::Output,
        bias: Option<&Self::Output>,
        eps: T,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        post_scalar: Option<fn(T) -> T>,
        post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        batchnorm_conv2d(
            self,
            kernels,
            mean,
            var,
            gamma,
            beta,
            bias,
            eps,
            steps,
            padding,
            dilation,
            post_scalar,
            post_vec,
        )
    }
}
